import torch
import os

def D_train(x, labels, G, D, D_optimizer, criterion, device, smoothing=0.1):
    D.zero_grad()

    x_real, y_real = x, torch.ones(x.shape[0], 1).to(device) * (1 - smoothing)
    labels = labels.to(device)

    D_output = D(x_real, labels)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    z = torch.randn(x.shape[0], 100).to(device)
    gen_labels = torch.randint(0, 10, (x.shape[0],)).to(device)
    x_fake, y_fake = G(z, gen_labels), torch.zeros(x.shape[0], 1).to(device) + smoothing

    D_output = D(x_fake, gen_labels)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()

def G_train(x, labels, G, D, G_optimizer, criterion, device, smoothing=0.1):
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    gen_labels = torch.randint(0, 10, (x.shape[0],)).to(device)
    y = torch.ones(x.shape[0], 1).to(device) * (1 - smoothing)

    G_output = G(z, gen_labels)
    D_output = D(G_output, gen_labels)
    G_loss = criterion(D_output, y)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))

def load_model(G, folder, device):
    ckpt = torch.load(os.path.join(folder, 'G.pth'), map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G