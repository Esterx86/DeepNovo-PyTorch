import loadspec
#import config
import torch
import numpy as np
#from torch import Generator
from torch.utils.data import Dataset, BatchSampler, Sampler, RandomSampler, SubsetRandomSampler, SequentialSampler

#rawdata = loadspec.load_data("./data/peaks.db.10k.mgf")
#rawdata = loadspec.load_data("/home/zhuyy/torch/peaks.db.mgf.test.dup")

class RawDataset(Dataset):
    def __init__(self, rawdata):
        super().__init__()
        self.rawdata = rawdata

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, i):
        return self.rawdata[i]

class SpecDataset(RawDataset):
    def __init__(self, rawdata):
        super().__init__(rawdata)
        self.data=[sample.preprocess() for sample in self.rawdata]

    def __getitem__(self, i):
        return self.data[i]
        # pep=self.rawdata[i]
        # padded,forward,backward=pep.pad_spectrum()
        # sequence_forward=pep.pad_seq(direction=0)
        # sequence_backward=pep.pad_seq(direction=1)
        # return padded,forward,backward,sequence_forward,sequence_backward,pep.neutral_mass()

class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size, buckets, drop_last=True, shuffle=True, generator=None):
        super().__init__(data_source)
        self.dataset = data_source
        self.buckets = dict((bucket_length, []) for bucket_length in buckets)
        self.batches = []
        for i, sample in enumerate(self.dataset):
            self.buckets[sample.padded_length].append(i)
        for bucket_length, bucket in self.buckets.items():
            sampler = SubsetRandomSampler(
                bucket, generator=generator) if shuffle else bucket
            batch = list(BatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last))
            self.batches.extend(batch)
            print(
                f'padded length: {bucket_length}\tbucket size: {len(bucket)}\tbatch count: {len(batch)}')
        print(f'total spectra: {len(self.dataset)}\ttotal batch: {len(self.batches)}')
        
        self.indices=RandomSampler(self.batches,generator=generator) if shuffle else SequentialSampler(self.batches)
        
    def __iter__(self):
        for batch_index in self.indices:
            yield [sample for sample in self.batches[batch_index]]

    def __len__(self):
        return len(self.batches)

def train_preprocess(batch):
    (padded_spectra,
     fragments_forward,
     fragments_backward,
     targets_forward,
     targets_backward,
     weights_forward,
     weights_backward) = zip(*[sample for sample in batch])
     #zip(*[sample.preprocess() for sample in batch])

    spectrum_batch=torch.from_numpy(np.stack(padded_spectra,axis=0)) #(128,30000)
    fragments_forward_batch=torch.from_numpy(np.stack(fragments_forward,axis=1)) #(L,128,26,8,10)
    fragments_backward_batch=torch.from_numpy(np.stack(fragments_backward,axis=1))
    target_forward_batch=torch.from_numpy(np.stack(targets_forward,axis=1)) #(L,128)
    target_backward_batch=torch.from_numpy(np.stack(targets_backward,axis=1))
    weight_forward_batch=torch.from_numpy(np.stack(weights_forward,axis=1)) #(L,128)
    weight_backward_batch=torch.from_numpy(np.stack(weights_backward,axis=1))

    return (spectrum_batch,
            fragments_forward_batch,
            fragments_backward_batch,
            target_forward_batch,
            target_backward_batch,
            weight_forward_batch,
            weight_backward_batch)

def test_preprocess(batch):
    (padded_spectra,
     spectra_forward,
     spectra_backward,
     fragments_forward,
     fragments_backward,
     targets_forward,
     targets_backward,
     weights_forward,
     weights_backward,
     neutral_mass,) = zip(*[sample.preprocess(open_loop=True) for sample in batch])
    
    #zip(*[sample.preprocess(original_spectrum=True) for sample in batch])

    spectrum_batch=torch.from_numpy(np.stack(padded_spectra,axis=0)) #(128,30000)
    spectrum_forward_batch=torch.from_numpy(np.stack(spectra_forward,axis=0))
    spectrum_backward_batch=torch.from_numpy(np.stack(spectra_backward,axis=0))
    fragments_forward_batch=torch.from_numpy(np.stack(fragments_forward,axis=1)) #(L,128,26,8,10)
    fragments_backward_batch=torch.from_numpy(np.stack(fragments_backward,axis=1))
    target_forward_batch=torch.from_numpy(np.stack(targets_forward,axis=1)) #(L,128)
    target_backward_batch=torch.from_numpy(np.stack(targets_backward,axis=1))
    weight_forward_batch=torch.from_numpy(np.stack(weights_forward,axis=1)) #(L,128)
    weight_backward_batch=torch.from_numpy(np.stack(weights_backward,axis=1))

    neutral_mass_batch=torch.tensor(neutral_mass)

    return (spectrum_batch,
            spectrum_forward_batch,
            spectrum_backward_batch,
            fragments_forward_batch,
            fragments_backward_batch,
            target_forward_batch,
            target_backward_batch,
            weight_forward_batch,
            weight_backward_batch,
            neutral_mass_batch,)


def inference_preprocess(batch):
    padded_spectra, spectra_forward, spectra_backward= zip(*[sample.pad_spectrum() for sample in batch])

    spectrum_batch=torch.from_numpy(np.stack(padded_spectra,axis=0)) #(128,30000)
    spectrum_forward_batch=torch.from_numpy(np.stack(spectra_forward,axis=0))
    spectrum_backward_batch=torch.from_numpy(np.stack(spectra_backward,axis=0))
    neutral_mass_batch = torch.tensor([sample.neutral_mass() for sample in batch])

    return spectrum_batch, spectrum_forward_batch, spectrum_backward_batch, neutral_mass_batch


# for bucket in config._buckets:
#     bucket_subset = Subset(train_set, [i for i in range(
#         len(train_set)) if train_set[i].padded_length == bucket])
#     print(f'bucket_length: {bucket} number: {len(bucket_subset)}')
# train_sampler = BucketSampler(data_source=train_set,
#                         batch_size=config.BATCH_SIZE, buckets=config._buckets)

#train_loader=DataLoader(train_set,batch_sampler=train_sampler,collate_fn=train_preprocess)


