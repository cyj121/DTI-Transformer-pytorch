import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset import get_train, get_val
from model.config_cnn import ConfigCNN
from model.config_drug import ConfigDrug
from model.config_classify import ConfigClassify
from model.model_classify import Classify


epochs = 100
lr = 0.001

config_drug = ConfigDrug()
config_cnn = ConfigCNN()
config_classify = ConfigClassify()

classify_net = Classify(config_drug, config_cnn, config_classify)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(classify_net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6, 8], gamma=0.1)

train_loader = get_train()
val_loader = get_val()

train_log_f = open('./log/train-loss.log', 'w')
val_log_f = open('./log/val-loss.log', 'w')

params_total = sum([param.nelement() for param in classify_net.parameters()]) / 1e6
print(params_total)
train_log_f.write('参数量 = {:.2f} M \n'.format(params_total))
val_log_f.write('参数量 = {:.2f} M \n'.format(params_total))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classify_net.to(device)

if not os.path.exists('./trained_model'):
    os.mkdir('./trained_model')
model_path = './trained_model'

best_val_loss = 999

for epoch in range(epochs):
    loader = iter(train_loader)
    classify_net.train()
    for step, data in enumerate(loader):
        drugs, proteins, labels = data
        drugs, proteins, labels = drugs.to(device), proteins.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classify_net(drugs, proteins)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=classify_net.parameters(), max_norm=1)
        optimizer.step()
        if step % 8 == 0:
            loss_str = f'epoch: {epoch} step: {step} train_loss: {loss}  lr: {lr_scheduler.get_last_lr()} \n'
            train_log_f.write(loss_str)
            print(loss_str)

    classify_net.eval()
    v_loader = iter(val_loader)
    val_losses = []
    with torch.no_grad():
        for val_step, val_data in enumerate(tqdm(v_loader)):  # tqdm()对每个batch和batch总数做的进度条
            drugs, proteins, labels = val_data
            drugs, proteins, labels = drugs.to(device), proteins.to(device), labels.to(device)
            outputs = classify_net(drugs, proteins)
            loss = criterion(outputs, labels)
            val_losses.append(loss)
    val_loss = torch.tensor(val_losses).mean().detach().item()  # ？
    val_loss_str = f'epoch: {epoch} val_loss: {val_loss}  lr: {lr_scheduler.get_last_lr()} \n'
    val_log_f.write(val_loss_str)
    print(val_loss_str)

    lr_scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(classify_net.state_dict(), os.path.join(model_path, 'best_model.pth'))

torch.save(classify_net.state_dict(), os.path.join(model_path, 'final_model.pth'))

train_log_f.close()
val_log_f.close()