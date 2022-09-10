import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.nn.utils.clip_grad import clip_grad_norm_

# from loadspec imp ort calc_fragments
import config
import random

masses = torch.tensor(config.masses_np, dtype=config.DTYPE, device=config.device)
mask = torch.tensor(config.mask, dtype=torch.long, device=config.device)
class Intensity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spectrum, pepmass, prefix_mass, direction):
        if direction == 0:
            FIRST_LABEL = config._GO
            LAST_LABEL = config._EOS
            # (N,) (26,) -> (N,26)
            candidate_b_mass = prefix_mass.unsqueeze(-1) + masses
            candidate_y_mass = pepmass.unsqueeze(-1) - candidate_b_mass
        elif direction == 1:
            FIRST_LABEL = config._EOS
            LAST_LABEL = config._GO
            candidate_y_mass = prefix_mass.unsqueeze(-1) + masses
            candidate_b_mass = pepmass.unsqueeze(-1) - candidate_y_mass

        # b-ions
        candidate_b_H2O = candidate_b_mass - config.mass_H2O
        candidate_b_NH3 = candidate_b_mass - config.mass_NH3
        candidate_b_plus2_charge1 = candidate_b_mass / 2

        # y-ions
        candidate_y_H2O = candidate_y_mass - config.mass_H2O
        candidate_y_NH3 = candidate_y_mass - config.mass_NH3
        candidate_y_plus2_charge1 = candidate_y_mass / 2

        ion_mass = torch.stack([
            candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1,
            candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1
        ], dim=-1)  # (N,26,8)

        candidate_intensity = torch.zeros(
            list(ion_mass.shape)+[config.WINDOW_SIZE], dtype=config.DTYPE, device=config.device)
        # (N,26,8,10)
        inf = torch.round(ion_mass*config.MS_RESOLUTION).long() - config.WINDOW_SIZE//2
        #sup = inf+config.WINDOW_SIZE  # (N,26,8)

        # indices = torch.logical_and(
        #     inf >= 0, sup <= config.MAX_MZ)  # BoolTensor(N,26,8)
        indices = torch.logical_and(inf >= 0, inf <= config.MAX_MZ-config.WINDOW_SIZE)       
        #indices = (inf >= 0) & sup <= (config.MAX_MZ)
        # slice = torch.stack(
        #     tuple(inf[indices]+i for i in range(config.WINDOW_SIZE)), dim=-1)
        # slice=torch.meshgrid(inf[indices],torch.arange(10,dtype=torch.long,device=config.device))
        slices = inf[indices].unsqueeze(dim=-1)+torch.arange(config.WINDOW_SIZE, device=config.device)
        # (K,10) the Tensor is flattened to (K,) then boolean indexed
        candidate_intensity[indices] = torch.take(spectrum, slices)
        #candidate_intensity.where((inf>=0)&(sup<=config.MAX_MZ),spectrum[inf:sup])

        candidate_intensity[:, FIRST_LABEL, :, :] = 0
        candidate_intensity[:, LAST_LABEL, :, :] = 0
        candidate_intensity[:, config._PAD, :, :] = 0
        # inf_view=inf.view(-1)
        # sup_view=sup.view(-1)
        # candi_view=candidate_intensity.view(-1,config.WINDOW_SIZE)
        # spectrum_view=spectrum.view(-1)

        # index=torch.logical_and(inf_view>=0,sup_view<=config.MZ_SIZE)
        # for i in range(inf_view.size(0)):
        #     if index[i]:
        #         candi_view[i]=spectrum_view[inf_view[i]:sup_view[i]]
        return candidate_intensity


class Spectra_CNN(nn.Module):  # (128,30000)→(128,512)
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, (1, 4), padding="same"),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 4, (1, 4), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((1, 6), stride=(1, 4), padding=(0, 1)),
            nn.Dropout(config.P_CONV)
        )
        self.fc = nn.Sequential(
            nn.Linear(config.MZ_SIZE, config.INPUT_SIZE),
            nn.ReLU(),
            nn.Dropout(config.P_DENSE)
        )

# initialization to-do

    def forward(self, x):
        x = x.view(-1, 1, 1, config.MZ_SIZE) # (128,1,1,30000)
        x = self.conv1(x)  # (128,4,1,30000)
        x = self.conv2(x)  # (128,4,1,7500)
        # x=F.dropout(x,config.P_CONV,self.training)
        x = x.view(-1,config.MZ_SIZE)  # (128,30000)
        encoded_spectrum = self.fc(x)  # (128,512)
        # encoded_spectrum=F.dropout(x,config.P_DENSE,self.training)
        return encoded_spectrum  # as initial cell state c_0


class Ion_CNN(nn.Module):  # (128,26,8,10)->(128,512)
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.VOCAB_SIZE, 64, (1, 3), padding="same"),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 2), padding="same"),
            nn.ReLU(),
            nn.Dropout(config.P_CONV),
            nn.MaxPool2d((1,3),stride=(1,2),padding=(0,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(config.ION_SIZE, config.INPUT_SIZE),
            nn.ReLU(),
            nn.Dropout(config.P_DENSE)
        )

    def forward(self, x):
        x = self.conv1(x)  # (128,64,8,10)
        x = self.conv2(x)  # (128,64,8,5)
        # x=F.dropout(x,config.P_CONV,self.training)
        x = x.view(-1,config.ION_SIZE)  # (128,2560)
        encoded_ions = self.fc(x)  # (128,512)
        # encoded_ions=F.dropout(x,0.5,self.training)
        return encoded_ions


class Seq_LSTM(nn.Module): # (128,512)->(128,512)
    def __init__(self) -> None:
        super().__init__()
        #self.embed = nn.Embedding(config.VOCAB_SIZE, config.INPUT_SIZE)
        self.lstm = nn.LSTMCell(
            input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE)
        self.dropout = nn.Dropout(config.P_DENSE)

    def forward(self, embedded, state):  # (128,512) (128,512) (128,512)
        #x = self.embed(embedded)  # (128,512)
        x = self.dropout(embedded)
        state = self.lstm(x, state)  # state=(h,c) (128,512) (128,512)
        decoder_output = self.dropout(state[0])
        return decoder_output, state  # projection later


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.encoder=spectra_CNN()
        self.dense_size = config.HIDDEN_SIZE*2

        self.decoder_cnn = Ion_CNN()
        self.decoder_lstm = Seq_LSTM()
        #self.embed = nn.Embedding(config.VOCAB_SIZE, config.INPUT_SIZE)
        #self.lstm=nn.LSTMCell(input_size=config.INPUT_SIZE,hidden_size=config.HIDDEN_SIZE)
        #self.lstm = nn.LSTM(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE)  # default: batch_first=False
        self.dense = nn.Sequential(
            nn.Linear(self.dense_size, config.VOCAB_SIZE),
            nn.ReLU(),
            nn.Dropout(config.P_DENSE),
            nn.LogSoftmax(dim=1),
        )

    def forward(self,
                embedded,  # (128,512) ground truth
                fragments,  # (128,26,8,10) windowed spectra
                state,
                #hidden,  # (1,512)
                #cell,  # (1,512)
                ):
        x1 = self.decoder_cnn(fragments)  # (128,512)
        x2, state = self.decoder_lstm(embedded, state)
        #x2 = self.embed(embedded) # (128,512)
        #x2= F.dropout(x2,p=config.P_DENSE,training=self.training)
        #state = self.lstm(x2, state) # h,c
        #x2 = state[0] # h (128,512)
        x = torch.concat((x1, x2), dim=1)  # (128,1024)
        logit = self.dense(x)  # (128,26)
        return logit, state


class Model(nn.Module):
    def __init__(self,direction=2,init=False,loss_func=nn.NLLLoss(reduction="none"),**kwargs):
        super().__init__()
        self.direction = direction
        self.encoder = Spectra_CNN()
        self.embed = nn.Embedding(config.VOCAB_SIZE, config.INPUT_SIZE)
        self.forward_decoder = Decoder() if direction == 0 or direction == 2 else None
        self.backward_decoder = Decoder() if direction == 1 or direction == 2 else None
        self.intens = Intensity()
        self.loss_func = loss_func

        if init:
            self.init_weights()
        #self.opt = SGD(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.opt=Adam(self.parameters(),**kwargs)

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
                nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu')
                nn.init.constant_(layer.bias,0.01)
            elif isinstance(layer,nn.Embedding):
                nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
            elif isinstance(layer,nn.LSTMCell):
                nn.init.orthogonal_(layer.weight_ih)
                nn.init.orthogonal_(layer.weight_hh)
                nn.init.constant_(layer.bias_ih,0.01)
                nn.init.constant_(layer.bias_hh,0.01)

    def init_hidden(self,batch_size=config.BATCH_SIZE):
        h0 = torch.zeros((batch_size, config.HIDDEN_SIZE), dtype=config.DTYPE, device=config.device)
        c0 = torch.zeros((batch_size, config.HIDDEN_SIZE), dtype=config.DTYPE, device=config.device)
        return h0, c0

    def decoding(
        self, 
        encoded_spectrum,
        fragments=None,
        sequence=None,
        weights=None,
        direction=0,):

        '''During training, sequence and weights must be specified,
        if use teacher forcing, fragments should be specified too,
        otherwise, original_spectra and pepmass should be specified.
        '''
        decoder=self.forward_decoder if direction==0 else self.backward_decoder
        
        # state0
        state0 = self.init_hidden(encoded_spectrum.shape[0])
        _, state = decoder.decoder_lstm(encoded_spectrum, state0)
        decode_length = sequence.size(dim=0)-1

        loss=0.0
        tp=0
        total_true=weights.sum()

        for i in range(decode_length):
            aa_input=sequence[i]
            fragment_input=fragments[i]
            embedded=self.embed(aa_input)
            logit, state = decoder(embedded, fragment_input, state) # (128,26)
            target = sequence[i+1]
            weight = weights[i+1]
            loss += self.loss_func(logit, target)*weight # (128,)
            _, output = logit.topk(1) # (128,1)
            output=output.squeeze().detach() # (128,)
            tp+=torch.where(output==target,weight,0.0).sum()

        loss_seq=loss/(weights.sum(dim=0)+1e-12) # (128,)
        loss_batch=loss_seq.mean() # (1,)

        return loss_batch,tp,total_true
        
    def val_decoding( #for visualization
        self, 
        encoded_spectrum,
        fragments=None,
        sequence=None,
        weights=None,
        original_spectra=None,
        pepmass=None,
        direction=0,
        teacher_forcing=True):

        '''During training, sequence and weights must be specified,
        if use teacher forcing, fragments should be specified too,
        otherwise, original_spectra and pepmass should be specified.
        '''
        decoder=self.forward_decoder if direction==0 else self.backward_decoder
        
        # state0
        state0 = self.init_hidden(encoded_spectrum.shape[0])
        _, state = decoder.decoder_lstm(encoded_spectrum, state0)
        decode_length = sequence.size(dim=0)-1
        #logits = []
        outputs = []
        loss=0.0

        for i in range(decode_length):
            aa_input=sequence[i]
            fragment_input=fragments[i]
            embedded=self.embed(aa_input)
            logit, state = decoder(embedded, fragment_input, state) # (128,26)
            target = sequence[i+1]
            weight = weights[i+1]
            loss += self.loss_func(logit, target)*weight # (128,)
            _, output = logit.topk(1) # (128,1)
            output=output.squeeze().detach() # (128,)
            outputs.append(output)

        outputs=torch.stack(outputs,dim=0) # (L-1,128)
        #targets=sequence[1:] # (L-1,128)
        tp,pp,total_true,total_pre=self.output_vs_target(
            outputs=outputs,targets=sequence[1:],weights=weights[1:])

        loss_seq=loss/(weights.sum(dim=0)+1e-12) # (128,)
        loss_batch=loss_seq.mean() # (1,)

        return loss_batch,tp,pp,total_true,total_pre,outputs # (L-1,128)

    def decoding_scheduled_sampling(
        self, 
        encoded_spectrum,
        fragments=None,
        sequence=None,
        weights=None,
        original_spectra=None,
        pepmass=None,
        direction=0,
        ratio=1.0):

        # if fragments == None or sequence == None or weights == None:
        #     teacher_forcing = False
        if direction == 0:
            decoder = self.forward_decoder
            start = config._GO
        else:
            decoder = self.backward_decoder
            start = config._EOS

        # state0
        state0 = self.init_hidden(encoded_spectrum.shape[0])
        _, state = decoder.decoder_lstm(encoded_spectrum, state0)
        decode_length = sequence.size(dim=0)-1
        #logits = []
        #outputs = []
        loss = 0.0
        tp = 0
        total_true = weights.sum()

        aa = torch.full_like(
            pepmass, start, dtype=config.DTYPE, device=config.device)
        prefix_mass = torch.zeros_like(
            pepmass, dtype=config.DTYPE, device=config.device)

        for i in range(decode_length):
            teacher = True if random.random() < ratio else False
            if teacher:
                aa_input = sequence[i]
                fragment_input = fragments[i]
            else:
                aa_input = aa
                fragment_input = self.intens(
                    original_spectra, pepmass, prefix_mass, direction=direction)
            embedded=self.embed(aa_input)
            logit, state = decoder(embedded, fragment_input, state)  # (128,26)
            target = sequence[i+1]
            weight = weights[i+1]
            #loss += self.criterion(logit, target)*weight # (128,)
            loss += self.loss_func(logit, target,)*weight  # (128,)
            _, output = logit.topk(1)  # (128,1)
            output = output.squeeze().detach()  # (128,)
            aa = output
            prefix_mass += masses[output]
            tp += torch.where(output == target, weight, 0.0).sum()
            #tp+=weight.where(output==target,0.0).sum()
            #outputs.append(output.squeeze().detach()) # (L-1,128)
            #logits.append(logit)  # (L-1,128,26)
            #logits = torch.stack(logits,dim=2)  # (128,26,L-1)
        loss_seq = loss/(weights.sum(dim=0)+1e-12)  # (128,)
        loss_batch = loss_seq.mean()  # (1,)
        # actual_lengths = weights[1:].sum(dim=0)+1e-12
        # loss_batch = torch.sum(loss/actual_lengths, dim=0)
        return loss_batch, tp, total_true

    @staticmethod
    def output_vs_target(outputs, targets, weights):
        output_weights = mask.take(outputs)
        condition=(outputs==targets)
        tp=torch.where(condition,weights,0.0).sum()
        pp=torch.where(condition,output_weights,0.0).sum()
        # fileter off pad
        total_true=weights.sum()
        total_pre=output_weights.sum()
        return tp,pp,total_true,total_pre

    def seq_loss_function(self, logits, targets, weights):
        '''
        logits: (N,26,L-1)
        targets: (N,L-1)
        weights: (N,L-1)
        '''
        losses= self.loss_func(logits, targets)*weights # (N,L-1)
        # for each sequence, loss is normalized over the valid length.
        loss_seq = losses.sum(dim=1)/(weights.sum(dim=1)+1e-12)
        loss_batch = loss_seq.mean(dim=None)
        return loss_batch # scalar

    def step(
            self,
            spectrum,
            fragments_forward=None,
            fragments_backward=None,
            sequence_forward=None,  # (L,128) tensor
            sequence_backward=None,
            target_weights_forward=None,
            target_weights_backward=None,
            direction=2):

        self.train()
        self.opt.zero_grad()
        encoded_spectrum = self.encoder(spectrum)
        loss_forward = 0.0
        loss_backward = 0.0
        hit_forward = 0
        hit_backward = 0
        total_forward = 0
        total_backward = 0

        if direction == 0 or direction == 2:

            loss_forward, hit_forward, total_forward = self.decoding(
                encoded_spectrum,
                fragments_forward,
                sequence_forward,
                target_weights_forward,
                direction=0,
                teacher_forcing=True
            )

        if direction == 1 or direction == 2:

            loss_backward, hit_backward, total_backward = self.decoding(
                encoded_spectrum,
                fragments_backward,
                sequence_backward,
                target_weights_backward,
                direction=1,
                teacher_forcing=True)

        total_loss = loss_forward+loss_backward
        total_loss /= (2 if direction == 2 else 1)
        hits = hit_forward+hit_backward
        totals = total_forward+total_backward

        total_loss.backward()
        grad_norm=clip_grad_norm_(self.parameters(),max_norm=config.MAX_GRADIENT_NORM)
        self.opt.step()
        #print(f'gradient norm: {grad_norm}')
        train_loss = total_loss.item()
        return train_loss, hits, totals

    def test_step(
            self,
            spectrum,
            spectrum_forward=None,
            spectrum_backward=None,
            fragments_forward=None,
            fragments_backward=None,
            sequence_forward=None,  # (L,128) tensor
            sequence_backward=None,
            target_weights_forward=None,
            target_weights_backward=None,
            pepmass=None,
            direction=2,
            ratio=1.0):

        self.eval()
        encoded_spectrum = self.encoder(spectrum)
        loss_forward = 0.0
        loss_backward = 0.0
        hit_forward = 0
        hit_backward = 0
        total_forward = 0
        total_backward = 0

        if direction == 0 or direction == 2:

            loss_forward, hit_forward, total_forward = self.decoding_scheduled_sampling(
                encoded_spectrum,
                fragments=fragments_forward,
                sequence=sequence_forward,
                weights=target_weights_forward,
                original_spectra=spectrum_forward,
                pepmass=pepmass,
                direction=0,
                ratio=ratio
            )

        if direction == 1 or direction == 2:

            loss_backward, hit_backward, total_backward = self.decoding_scheduled_sampling(
                encoded_spectrum,
                fragments=fragments_backward,
                sequence=sequence_backward,
                weights=target_weights_backward,
                original_spectra=spectrum_backward,
                pepmass=pepmass,
                direction=1,
                ratio=ratio
            )

        total_loss = loss_forward+loss_backward
        hits = hit_forward+hit_backward
        totals = total_forward+total_backward

        val_loss = total_loss.item()
        val_loss /= (2 if direction == 2 else 1)
        return val_loss, hits, totals

    def train_step( # support visualization
            self,
            spectrum,
            fragments_forward=None,
            fragments_backward=None,
            sequence_forward=None,  # (L,128) tensor
            sequence_backward=None,
            target_weights_forward=None,
            target_weights_backward=None,
            direction=2):

        self.train()
        self.opt.zero_grad()
        encoded_spectrum = self.encoder(spectrum)
        loss_forward = 0.0
        loss_backward = 0.0
        tp_forward=0
        tp_backward=0
        pp_forward=0
        pp_backward=0
        total_true_forward=0
        total_true_backward=0
        total_pre_forward=0
        total_pre_backward=0
        outputs_forward=None
        outputs_backward=None

        if direction == 0 or direction == 2:
            (loss_forward,
             tp_forward, pp_forward,
             total_true_forward, total_pre_forward,
             outputs_forward) = self.val_decoding(
                encoded_spectrum,
                fragments_forward,
                sequence_forward,
                target_weights_forward,
                direction=0,
                teacher_forcing=True
            )

        if direction == 1 or direction == 2:

            (loss_backward,
             tp_backward, pp_backward,
             total_true_backward, total_pre_backward,
             outputs_backward) = self.val_decoding(
                encoded_spectrum,
                fragments_backward,
                sequence_backward,
                target_weights_backward,
                direction=1,
                teacher_forcing=True
            )

        total_loss = loss_forward+loss_backward
        tp = tp_forward+tp_backward
        pp = pp_forward+pp_backward
        total_true = total_true_forward+total_true_backward
        total_pre = total_pre_forward+total_pre_backward

        total_loss/= (2 if direction == 2 else 1)

        total_loss.backward()
        grad_norm=clip_grad_norm_(self.parameters(),max_norm=config.MAX_GRADIENT_NORM)
        self.opt.step()

        train_loss = total_loss.item()

        return (
            train_loss,
            tp, pp,
            total_true, total_pre,
            outputs_forward, outputs_backward
        )

    def val_step( # support visualization
            self,
            spectrum,
            fragments_forward=None,
            fragments_backward=None,
            sequence_forward=None,  # (L,128) tensor
            sequence_backward=None,
            target_weights_forward=None,
            target_weights_backward=None,
            direction=2):

        self.eval()
        encoded_spectrum = self.encoder(spectrum)
        loss_forward = 0.0
        loss_backward = 0.0
        tp_forward=0
        tp_backward=0
        pp_forward=0
        pp_backward=0
        total_true_forward=0
        total_true_backward=0
        total_pre_forward=0
        total_pre_backward=0
        outputs_forward=None
        outputs_backward=None

        if direction == 0 or direction == 2:
            (loss_forward,
             tp_forward, pp_forward,
             total_true_forward, total_pre_forward,
             outputs_forward) = self.val_decoding(
                encoded_spectrum,
                fragments_forward,
                sequence_forward,
                target_weights_forward,
                direction=0,
                teacher_forcing=True
            )

        if direction == 1 or direction == 2:

            (loss_backward,
             tp_backward, pp_backward,
             total_true_backward, total_pre_backward,
             outputs_backward) = self.val_decoding(
                encoded_spectrum,
                fragments_backward,
                sequence_backward,
                target_weights_backward,
                direction=1,
                teacher_forcing=True
            )

        total_loss = loss_forward+loss_backward
        tp = tp_forward+tp_backward
        pp = pp_forward+pp_backward
        total_true = total_true_forward+total_true_backward
        total_pre = total_pre_forward+total_pre_backward

        total_loss/= (2 if direction == 2 else 1)
        val_loss = total_loss.item()

        return (
            val_loss,
            tp, pp,
            total_true, total_pre,
            outputs_forward, outputs_backward
        )

class Focalloss(nn.NLLLoss):
    def __init__(self,weight=None,gamma=None,size_average=None,reduction='mean'):
        super(Focalloss,self).__init__(weight, size_average, reduction)
        self.gamma=gamma
    def forward(self, inputs, targets) :
        preds=inputs.exp()
        if self.gamma==None:
            pass
        else:
            preds=(1.0-inputs).pow(self.gamma)*inputs.log()
        return self.loss_func(preds, targets, weight=self.weight, reduction=self.reduction)