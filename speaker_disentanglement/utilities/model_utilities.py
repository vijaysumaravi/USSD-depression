import torch
import torch.nn as nn
import pickle as pkl
import numpy as np

def create_tensor_data(x, cuda):
    """
    Converts the data from numpy arrays to torch tensors

    Inputs
        x: The input data
        cuda: Bool - Set to true if using the GPU

    Output
        x: Data converted to a tensor
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def read_speaker_embeddings(embd_path):
    """
    Reads the speaker embeddings from the path

    Inputs
        embd_path: The path to the speaker embeddings

    Output
        embeddings: The speaker embeddings
    """
    with open(embd_path, 'rb') as f:
        embeddings = pkl.load(f)

    return embeddings

def get_batch_speaker_embeddings(batch_folders,speaker_embeddings):

    # return embeddings for specific speakers as np array
    batch_embeddings = []
    for folder in batch_folders:
        batch_embeddings.append(speaker_embeddings[str(folder)])
    batch_embeddings = np.array(batch_embeddings)

    return torch.from_numpy(batch_embeddings)

def calculate_loss(prediction, target, cw=None, gender=True):
    """
    With respect to the final layer of the model, calculate the loss of the
    model.

    Inputs
        prediction: The output of the model
        target: The relative label for the output of the model
        cw: torch.Tensor - The class weights for the dataset
        gender: bool set True if splitting data according to gender

    Output
        loss: The BCELoss or NLL_Loss
    """
    if gender:
        if target.shape[0] != cw.shape[0]:
            fem_nd_w, fem_d_w, male_nd_w, male_d_w = cw
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            two_ind = (target == 2).nonzero().reshape(-1)
            three_ind = (target == 3).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, fem_nd_w[0])
            class_weights.scatter_(0, one_ind, fem_d_w[0])
            class_weights.scatter_(0, two_ind, male_nd_w[0])
            class_weights.scatter_(0, three_ind, male_d_w[0])
            cw = class_weights.reshape(-1, 1)
        target = target % 2
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
    else:
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
        if target.shape[0] != cw.shape[0]:
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, cw[0])
            class_weights.scatter_(0, one_ind, cw[1])
            cw = class_weights.reshape(-1, 1)

    if prediction.dim() == 1:
        prediction = prediction.view(-1, 1)

    bceloss = nn.BCELoss(weight=cw)
    loss = bceloss(prediction, target.float().view(-1, 1))

    return loss


def calculate_speaker_loss(prediction, target):
    """
    With respect to the speaker prediction layer of the model, calculate the loss of the
    model.

    Inputs
        prediction: The output of the model
        target: The relative label for the output of the model

    Output
        loss: The CELoss
    """
    

    celoss = nn.CrossEntropyLoss()
    loss = celoss(prediction, target)
    return loss


def calculate_speaker_loss_LE(prediction ,target):
    mseloss = nn.MSELoss(reduction='sum')
    projection = nn.Softmax()
    prediction = projection(prediction)
    loss = mseloss(prediction, target)# 20 x 107
    loss = loss / 20 #batch size
    return loss

#def calculate_speaker_loss_LE_1(prediction, target):
#    import pdb
#    pdb.set_trace()
#    celoss = nn.CrossEntropyLoss()
#    loss = celoss(prediction, target)
#    return loss

def calculate_speaker_loss_LE_KL(prediction, target):
    projection = nn.LogSoftmax()
    KL_loss = nn.KLDivLoss(reduction='batchmean')
    prediction = projection(prediction)
    loss = KL_loss(prediction, target)
    return loss

def calculate_speaker_loss_LE_KL_1(prediction, target):
    import pdb
    KL_loss = nn.KLDivLoss(reduction='batchmean')
    loss = KL_loss(prediction, target)
    return loss

def calculate_speaker_loss_dist(model_embd_output, spk_embd):

    # calculate cosine similarity between model output and speaker embedding for each speaker
    num_spk = model_embd_output.shape[0]
    predicted_similarity = torch.zeros((num_spk,num_spk))
    target_similarity = torch.zeros((num_spk,num_spk)) + torch.randn(num_spk,num_spk) * 1e-8
    for i in range(num_spk):
        for j in range(num_spk):
            predicted_similarity[i,j] = nn.functional.cosine_similarity(model_embd_output[i],spk_embd[j],dim=0)
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_similarity, target_similarity)
    return loss

def calculate_speaker_loss_dist_no_noise(model_embd_output, spk_embd):

    # calculate cosine similarity between model output and speaker embedding for each speaker
    num_spk = model_embd_output.shape[0]
    predicted_similarity = torch.zeros((num_spk,num_spk))
    target_similarity = torch.zeros((num_spk,num_spk)) #+ torch.randn(num_spk,num_spk) * 1e-8
    for i in range(num_spk):
        for j in range(num_spk):
            predicted_similarity[i,j] = nn.functional.cosine_similarity(model_embd_output[i],spk_embd[j],dim=0)
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_similarity, target_similarity)
    return loss