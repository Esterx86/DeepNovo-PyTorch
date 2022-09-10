# distutils: language=c++
# cython: c_string_type=str, c_string_encoding=ascii

import numpy as np
cimport numpy as np
np.import_array()
cimport cython
from libc.math cimport lrint,fmax
from libcpp.algorithm cimport copy
from libc.stdlib cimport malloc, free
from libc.string cimport strcspn
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
#from libcpp.random cimport default_random_engine,uniform_real_distribution

import config
cdef int charge=config.CHARGE
cdef double max_mz=config.MAX_MZ
cdef int max_len=config.MAX_LEN
cdef double resolution=config.MS_RESOLUTION
cdef int window_size=config.WINDOW_SIZE
cdef int mz_size=int(max_mz*resolution)

cdef int  _PAD=config._PAD
cdef int  _GO=config._GO
cdef int  _EOS=config._EOS

cdef double mass_H=config.mass_H
cdef double mass_H2O=config.mass_H2O
cdef double mass_NH3=config.mass_NH3
cdef double mass_N_terminus=config.mass_N_terminus
cdef double mass_C_terminus=config.mass_C_terminus

cdef map[string, int] vocab=config.vocab
cdef int vocab_size=vocab.size()
cdef map[int, double] masses=config.masses
cdef np.ndarray masses_np=config.masses_np

cdef int num_ion=config.num_ion
cdef vector[int] buckets=config._buckets


cdef extern from "fileio.cpp":
    cdef cppclass Spectrum:
        char title[256]
        double pepmass
        int charge
        double retsec
        char sequence[256]
        int length
        int peak
        double* mz
        double* intensity
    cdef vector[Spectrum*] parse_file(const char* path)
    cdef void clear_spectrum(Spectrum* spectrum)
    #cdef void clear_spectra(vector[Spectrum]& spectra)
    
cdef class Peptide:
    cdef Spectrum* _ptr
    cdef bint _ptr_own
    cdef bint _modified,_unknownmod
    cdef vector[int] _encoded_seq
    cdef int _padded_length
    def __cinit__(self):
        # self._ptr=NULL
        self._ptr_own=False
        self._modified=False
        self._unknownmod=False
    def __init__(self):
        raise TypeError("This class cannot be instantiated directly.")
        
    def __dealloc__(self):
        if (self._ptr is not NULL) and (self._ptr_own is True):
            clear_spectrum(self._ptr)# free mz, intensity
            del self._ptr# free Spectrum
            self._ptr=NULL
            self._ptr_own=False

    @staticmethod
    cdef Peptide from_spec(Spectrum* spec,bint owner=False):
        cdef Peptide pep=Peptide.__new__(Peptide)
        pep._ptr=spec
        pep._ptr_own=owner
        pep._modified=False
        pep._unknownmod=False

        pep._encoded_seq=pep.encoding_seq()
        for i in buckets:
            if pep._ptr.length + 2 <= i:
                pep._padded_length = i
                break
            
        return pep

    @property
    def pepmass(self):
        return self._ptr.pepmass if self._ptr is not NULL else None
    @property
    def charge(self):
        return self._ptr.charge if self._ptr is not NULL else None
    @property
    def length(self):
        return self._ptr.length if self._ptr is not NULL else None
    @property
    def sequence(self):
        return self._ptr.sequence if self._ptr is not NULL else None
    @property
    def seq_code(self):
        return list(self._encoded_seq) if self._unknownmod is False else None
    @property
    def padded_length(self):
        return self._padded_length if self._ptr.sequence is not NULL else None
    @property
    def ismodified(self):
        return self._modified if self._ptr.sequence is not NULL else None

    cpdef double neutral_mass(self):
        return self._ptr.pepmass-self._ptr.charge*mass_H

    cdef vector[int] encoding_seq(self):
        cdef char* pt=self._ptr.sequence
        cdef vector[int] codes
        codes.reserve(24)
        #codes.push_back(vocab[b'_GO'])
        codes.push_back(_GO)
        cdef int span
        while pt[0]:
            if (pt+1)[0]==b'(':
                self._modified=True
                span=strcspn(pt,')')+1
                try:
                    codes.push_back(vocab[string(pt,span)])
                except:
                    self._unknownmod=True
                    codes.clear()
                    return codes
                pt+=span
            else:
                codes.push_back(vocab[string(pt,1)])
                pt+=1
        #codes.push_back(vocab[b'_EOS'])
        codes.push_back(_EOS)
        return codes

    cpdef vector[int] pad_seq(self,int direction=0):

        cdef vector[int] padded_seq=vector[int](self._padded_length,_PAD)
        
        if direction==0:
            copy(self._encoded_seq.begin(),self._encoded_seq.end(),padded_seq.begin())
        else:
            copy(self._encoded_seq.rbegin(),self._encoded_seq.rend(),padded_seq.begin())

        return padded_seq

    # @cython.infer_types(True)
    # cpdef vector[int] target_and_weight(self,int direction=0):

    #     cdef vector[int] padded_seq=vector[int](self._padded_length,vocab[b'_PAD'])
    #     cdef vector[int] target_weight=vector[int](self._padded_length,1)
    #     #cdef vector[int].iterator seq,weight,first,last
    #     seq=padded_seq.begin()
    #     weight=target_weight.begin()

    #     if direction==0:
    #         first=self._encoded_seq.begin()
    #         last=self._encoded_seq.end()
    #     elif direction==1:
    #         first=self._encoded_seq.rbegin()
    #         last=self._encoded_seq.rend()

    #     while first!=last-1:
    #         deref(seq)=deref(first)
    #         target=deref(first+1)
    #         if target==vocab[b'_PAD'] or target==vocab[b'_GO'] or target==vocab[b'_EOS']:
    #             deref(weight)=0
    #         first+=1
    #         seq+=1
    #         weight+=1
    #     deref(seq)=deref(first)
    #     #seq[0]=first[0]
    #     return padded_seq

    cpdef bint verify(self, bint mass=True, bint length=True, bint unknownmod=True):
        cdef bint crit1 = self._ptr.mz[self._ptr.peak-1] < max_mz if mass else True
        cdef bint crit2 = self._ptr.length < max_len if length else True
        cdef bint crit3 = not self._unknownmod if unknownmod else True
        return crit1 and crit2 and crit3
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef pad_spectrum(self):
        padded=np.zeros(mz_size,dtype=np.float64)
        cdef double[::1] padded_view=padded

        cdef int* complement_index_view=<int*>malloc(self._ptr.peak*sizeof(int))
        #cdef double* intensity_view=<double*>malloc(self._ptr.peak*sizeof(double))
        #memcpy(intensity_view,self._ptr.intensity,self._ptr.peak*sizeof(double))

        cdef int i,index,cindex
        cdef double maxint=0.0, mass
        cdef double pepmass=self.neutral_mass()
        
        for i in range(self._ptr.peak):
            maxint=fmax(maxint,self._ptr.intensity[i])
        for i in range(self._ptr.peak):
            mass=self._ptr.mz[i]-charge*mass_H # b-ion: backbone + proton, y-ion: backbone + H2O + proton
            index=lrint(mass*resolution)
            cindex=lrint((pepmass-mass)*resolution)
            complement_index_view[i]=cindex
            # original method: normalize intensity, then pick the maximium
            padded_view[index]=fmax(padded_view[index],self._ptr.intensity[i]/maxint)
            # my idea
            # padded_view[index]+=self._ptr.intensity[i]/maxint
        #cdef double maxint=padded.max()

        forward=np.copy(padded)
        backward=np.copy(padded)

        for i in range(self._ptr.peak):
            cindex=complement_index_view[i]
            if cindex>0:
                padded_view[cindex]+=self._ptr.intensity[i]/maxint
        free(complement_index_view)

        # entire peptide
        cdef int pep_index = lrint(pepmass * resolution)
        forward[pep_index] = 1.0
        backward[pep_index] = 1.0

        # N-terminus(proton, neutralized to nothing)
        cdef double mass_N = mass_N_terminus - mass_H
        cdef int n_index = lrint(mass_N * resolution)
        padded[n_index] = 1.0
        # peptide as y-ion(peptide + proton, neutralized to peptide)
        cdef double mass_Y = pepmass - mass_N
        cdef int y_index = lrint(mass_Y * resolution)
        padded[y_index] = 1.0
        backward[y_index] = 1.0

        # C-termius(hydroxyl, neutralized to H2O)
        cdef double mass_C = mass_C_terminus + mass_H
        cdef int c_index = lrint(mass_C * resolution)
        padded[lrint(c_index)] = 1.0
        # peptide as b-ion(peptide + proton - H2O, neutralized to peptide - H2O)
        cdef double mass_B = pepmass - mass_C
        cdef int b_index = lrint(mass_B * resolution)
        padded[b_index] = 1.0
        forward[b_index] = 1.0

        return padded,forward,backward

    cpdef pad_spectrum_np(self):
        cdef np.npy_intp dim = np.PyArray_PyIntAsIntp(mz_size)
        cdef np.ndarray padded = np.PyArray_ZEROS(1,&dim,type=np.NPY_FLOAT64,fortran=0)
        #padded=np.zeros(mz_size,dtype=np.float64)
        cdef double[::1] padded_view=padded

        #cdef np.ndarray[int] complement_index=np.zeros(shape=(self._ptr.peak,),dtype=np.int32)
        #complement_index=np.zeros(shape=(self._ptr.peak,),dtype=np.int32)
        #cdef int[::1] complement_index_view=complement_index
        cdef int* complement_index_view=<int*>malloc(self._ptr.peak*sizeof(int))
        #cdef int[:] complement_index_view=<int[:self._ptr.peak]>complement_index
        #complement_index_view[...]=0
        #cdef double maxint=*max_element(self._ptr.intensity, self._ptr.intensity + self._ptr.peak)

        cdef int i,index,cindex
        cdef double maxint=0.0, mass
        cdef double pepmass=self.neutral_mass()
        
        for i in range(self._ptr.peak):
            maxint=fmax(maxint,self._ptr.intensity[i])

        for i in range(self._ptr.peak):
            #maxint=fmax(maxint,self._ptr.intensity[i])
            
            mass=self._ptr.mz[i]-charge*mass_H # b-ion: backbone + proton, y-ion: backbone + H2O + proton
            index=lrint(mass*resolution)
            cindex=lrint((pepmass-mass)*resolution)
            complement_index_view[i]=cindex
            # original method: normalize intensity, then pick the maximium
            padded_view[index]=fmax(padded_view[index],self._ptr.intensity[i]/maxint)
            # my idea
            # padded_view[index]+=self._ptr.intensity[i]/maxint
        #cdef double maxint=padded.max()
        #padded=padded/maxint
        cdef np.ndarray forward=np.PyArray_Copy(padded)
        #forward=np.copy(padded)
        cdef np.ndarray backward=np.PyArray_Copy(padded)
        #backward=np.copy(padded)

        for i in range(self._ptr.peak):
            cindex=complement_index_view[i]
            if cindex>0:
                padded_view[cindex]+=self._ptr.intensity[i]/maxint
        #free(complement_index_view)

        # entire peptide
        cdef int pep_index = lrint(pepmass * resolution)
        forward[pep_index] = 1.0
        backward[pep_index] = 1.0

        # N-terminus(proton, neutralized to nothing)
        cdef double mass_N = mass_N_terminus - mass_H
        cdef int n_index = lrint(mass_N * resolution)
        padded[n_index] = 1.0
        # peptide as y-ion(peptide + proton, neutralized to peptide)
        cdef double mass_Y = pepmass - mass_N
        cdef int y_index = lrint(mass_Y * resolution)
        padded[y_index] = 1.0
        backward[y_index] = 1.0

        # C-termius(hydroxyl, neutralized to H2O)
        cdef double mass_C = mass_C_terminus + mass_H
        cdef int c_index = lrint(mass_C * resolution)
        padded[lrint(c_index)] = 1.0
        # peptide as b-ion(peptide + proton - H2O, neutralized to peptide - H2O)
        cdef double mass_B = pepmass - mass_C
        cdef int b_index = lrint(mass_B * resolution)
        padded[b_index] = 1.0
        forward[b_index] = 1.0

        return padded,forward,backward

    cpdef calc_fragments(self,double[:] spectrum,double prefix_mass,int direction):
        if direction == 0:
            FIRST_LABEL = _GO
            LAST_LABEL = _EOS
            candidate_b_mass = prefix_mass + masses_np
            candidate_y_mass = self.neutral_mass() - candidate_b_mass
        else:
            FIRST_LABEL = _EOS
            LAST_LABEL = _GO
            candidate_y_mass = prefix_mass + masses_np
            candidate_b_mass = self.neutral_mass() - candidate_y_mass

        # b-ions
        candidate_b_H2O = candidate_b_mass - mass_H2O
        candidate_b_NH3 = candidate_b_mass - mass_NH3
        candidate_b_plus2_charge1 = candidate_b_mass / 2
        # ((candidate_b_mass + 2 * mass_H) / 2 - mass_H)
        # y-ions
        candidate_y_H2O = candidate_y_mass - mass_H2O
        candidate_y_NH3 = candidate_y_mass - mass_NH3
        candidate_y_plus2_charge1 = candidate_y_mass / 2
        # ((candidate_y_mass + 2 * mass_H) / 2 - mass_H)
        ion_mass = np.stack([
            candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1,
            candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1
            ],axis=-1)#(26,8)
            #how about without copying?

        candidate_intensity = np.zeros(
            shape=(vocab_size,num_ion,window_size), #(26,8,10)
            dtype=np.float64)

        cdef double [:,::1] ion_mass_view=ion_mass       
        cdef double [:,:,::1] candidate_intensity_view = candidate_intensity
        cdef int row,col,sup,inf
        for row in range(vocab_size):
            for col in range(num_ion):
                inf=lrint(ion_mass_view[row,col]*resolution)-window_size//2
                sup=inf+window_size
                if inf>=0 and sup<=max_mz:
                    candidate_intensity_view[row,col,:]=spectrum[inf:sup]
        candidate_intensity_view[FIRST_LABEL,:,:]=0
        candidate_intensity_view[LAST_LABEL,:,:]=0
        candidate_intensity_view[_PAD,:,:]=0
        return candidate_intensity

    # cpdef calc_fragments_fast(self,double[:] spectrum,double prefix_mass,int direction):
    #     if direction == 0:
    #         FIRST_LABEL = vocab[b'_GO']
    #         LAST_LABEL = vocab[b'_EOS']
    #         candidate_b_mass = prefix_mass + masses_np
    #         candidate_y_mass = self.neutral_mass() - candidate_b_mass
    #     else:
    #         FIRST_LABEL = vocab[b'_EOS']
    #         LAST_LABEL = vocab[b'_GO']
    #         candidate_y_mass = prefix_mass + masses_np
    #         candidate_b_mass = self.neutral_mass() - candidate_y_mass


    #     ion_mass = np.stack([
    #         candidate_b_mass,
    #         candidate_b_mass - mass_H2O,
    #         candidate_b_mass - mass_NH3,
    #         candidate_b_mass / 2,
    #         candidate_y_mass,
    #         candidate_y_mass - mass_H2O,
    #         candidate_y_mass - mass_NH3,
    #         candidate_y_mass / 2
    #         ],axis=-1)#(26,8)

    #     candidate_intensity = np.zeros(
    #         shape=(vocab_size,num_ion,window_size), #(26,8,10)
    #         dtype=np.float64)

    #     cdef double [:,::1] ion_mass_view=ion_mass       
    #     cdef double [:,:,::1] candidate_intensity_view = candidate_intensity
    #     cdef int row,col,sup,inf
    #     for row in range(vocab_size):
    #         for col in range(num_ion):
    #             inf=lrint(ion_mass_view[row,col]*resolution)-window_size//2
    #             sup=inf+window_size
    #             if inf>=0 and sup<=max_mz:
    #                 candidate_intensity_view[row,col,:]=spectrum[inf:sup]
    #     candidate_intensity_view[FIRST_LABEL,:,:]=0
    #     candidate_intensity_view[LAST_LABEL,:,:]=0
    #     candidate_intensity_view[vocab[b'PAD'],:,:]=0
    #     return candidate_intensity
    
    def preprocess(self,int direction=2,bint open_loop=False):
        cdef double prefix_mass
        cdef vector[int] sequence_forward,sequence_backward
        cdef vector[double] weight_forward,weight_backward
        cdef list candidate_forward = [],candidate_backward = []

        padded_spectrum,spectrum_forward,spectrum_backward=self.pad_spectrum()
        if direction==0 or direction==2:
            prefix_mass=0.0
            sequence_forward=self.pad_seq(direction=0)
            weight_forward=vector[double](self._padded_length,1.0)
            for i in range(self._padded_length):
                prefix_mass+=masses[sequence_forward[i]]
                candidate_forward.append(
                    self.calc_fragments(spectrum=spectrum_forward,prefix_mass=prefix_mass,direction=0))
                if i<self._padded_length-1:
                    target=sequence_forward[i+1]
                if (i==self._padded_length-1 
                    or target==_PAD
                    or target==_GO
                    or target==_EOS):
                    weight_forward[i]=0.0

        if direction==1 or direction==2:
            prefix_mass=0.0
            sequence_backward=self.pad_seq(direction=1)
            weight_backward=vector[double](self._padded_length,1.0)
            for i in range(self._padded_length):
                prefix_mass+=masses[sequence_backward[i]]
                candidate_backward.append(
                    self.calc_fragments(spectrum=spectrum_backward,prefix_mass=prefix_mass,direction=1))
                if i<self._padded_length-1:
                    target=sequence_backward[i+1]
                if (i==self._padded_length-1 
                    or target==_PAD
                    or target==_GO
                    or target==_EOS):
                    weight_backward[i]=0.0
            # for aa in padded_seq:
            #     prefix_mass+=masses[aa]
            #     fragments_backward.append(
            #         self.calc_fragments(prefix_mass=prefix_mass,spectrum=backward,direction=1))

        fragments_forward=np.array(candidate_forward,dtype=np.float64)# (L,26,8,10)
        fragments_backward=np.array(candidate_backward,dtype=np.float64)
        target_forward=np.array(sequence_forward,dtype=np.int32) #(L,)
        target_backward=np.array(sequence_backward,dtype=np.int32)
        target_weight_forward=np.array(weight_forward,dtype=np.float64)
        target_weight_backward=np.array(weight_backward,dtype=np.float64)

        if open_loop:
            return (
                padded_spectrum,
                spectrum_forward,
                spectrum_backward,
                fragments_forward,
                fragments_backward,
                target_forward,
                target_backward,
                target_weight_forward,
                target_weight_backward,
                self.neutral_mass(),
            )
        else:
            return (
                padded_spectrum,
                fragments_forward,
                fragments_backward,
                target_forward,
                target_backward,
                target_weight_forward,
                target_weight_backward,
                )


def calc_fragments(double[:] spectrum, double pepmass, double prefix_mass,int direction):
    if direction == 0:
        FIRST_LABEL = _GO
        LAST_LABEL = _EOS
        candidate_b_mass = prefix_mass + masses_np
        candidate_y_mass = pepmass - candidate_b_mass
    else:
        FIRST_LABEL = _EOS
        LAST_LABEL = _GO
        candidate_y_mass = prefix_mass + masses_np
        candidate_b_mass = pepmass - candidate_y_mass

        # b-ions
        # candidate_b_H2O = candidate_b_mass - mass_H2O
        # candidate_b_NH3 = candidate_b_mass - mass_NH3
        # candidate_b_plus2_charge1 = candidate_b_mass / 2

        # # y-ions
        # candidate_y_H2O = candidate_y_mass - mass_H2O
        # candidate_y_NH3 = candidate_y_mass - mass_NH3
        # candidate_y_plus2_charge1 = candidate_y_mass / 2

    ion_mass = np.stack([
        candidate_b_mass,
        candidate_b_mass - mass_H2O,
        candidate_b_mass - mass_NH3,
        candidate_b_mass / 2,
        candidate_y_mass,
        candidate_y_mass - mass_H2O,
        candidate_y_mass - mass_NH3,
        candidate_y_mass / 2
        ],axis=-1)#(26,8)
        #how about without copying?

    candidate_intensity = np.zeros(
        shape=(vocab_size,num_ion,window_size), #(26,8,10)
        dtype=np.float64)

    cdef double [:,::1] ion_mass_view=ion_mass       
    cdef double [:,:,::1] candidate_intensity_view = candidate_intensity
    cdef int row,col,sup,inf
    for row in range(vocab_size):
        for col in range(num_ion):
            inf=lrint(ion_mass_view[row,col]*resolution)-window_size//2
            sup=inf+window_size
            if inf>=0 and sup<=max_mz:
                candidate_intensity_view[row,col,:]=spectrum[inf:sup]
    candidate_intensity_view[FIRST_LABEL,:,:]=0
    candidate_intensity_view[LAST_LABEL,:,:]=0
    candidate_intensity_view[_PAD,:,:]=0
    return candidate_intensity # (26,8,10)

def load_data(const char* path,**kwargs):
    cdef vector[Spectrum*] spectra=parse_file(path)
    cdef Spectrum* spec
    cdef list rawdata=[]
    for spec in spectra:
        pep=Peptide.from_spec(spec,owner=True)
        if pep.verify(**kwargs):
            rawdata.append(pep)
    print(f'Total spectra: {spectra.size()} Valid spectra: {len(rawdata)}')
    return rawdata

def load_bucket_data(const char* path,**kargs):
    cdef vector[Spectrum*] spectra=parse_file(path)
    cdef Spectrum* spec
    cdef dict rawdata=dict((bucket,[]) for bucket in buckets)
    for spec in spectra:
        pep=Peptide.from_spec(spec,owner=True)
        if pep.verify(**kargs):
            rawdata[pep.padded_length].append(pep)
    return rawdata

# def load_banlanced_data(const char* path, int max_spectra_count):
#     cdef vector[Spectrum*] spectra=parse_file(path)
#     cdef Spectrum* spec
#     cdef map[char*,int] table
#     cdef list rawdata=[]

#     for spec in spectra:
#         pep=Peptide.from_spec(spec,owner=True)
#         if pep.verify():
#             if table.find(pep._ptr.sequence):
#                 rawdata.append(pep)
#             else:
#                 dies=
#                 table[pep._ptr.sequence]=