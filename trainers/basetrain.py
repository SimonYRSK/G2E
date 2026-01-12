from tqdm import tqdm
import torch

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, scheduler, epochs, device, beta):
        self.model = model
        self.trainlo = train_loader
        self.testlo = test_loader
        self.opt = optimizer
        self.sch = scheduler
        self.epochs = epochs
        self.device = device
        self.beta = beta

    def cal_losses(self, x, y, mu, logvar):
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        recon_loss = torch.mean((x - y) ** 2)

        return kl_loss, recon_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(self.trainlo, desc=f"Epoch {epoch+1}/{self.epochs}")

        for batch_idx, (x, y, t) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            t = t.to(self.device)

            x_recon, mu, log_var = self.model(x)
            kl_loss, recon_loss = self.cal_losses(x_recon, y, mu, log_var)
            loss = kl_loss * self.beta + recon_loss

            total_loss += loss
            total_recon_loss += recon_loss 
            total_kl_loss += kl_loss

            self.opt.zero_grad()
            loss.backward()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
            })

            self.opt.step()
        
        self.sch.step()

        avg_loss = total_loss / len(self.trainlo)
        avg_recon = total_recon_loss / len(self.trainlo)
        avg_kl = total_kl_loss / len(self.trainlo)
        print(f"\nEpoch {epoch+1} 训练集平均:")
        print(f"总损失={avg_loss:.5f}, 重建={avg_recon:.5f}, 散度={avg_kl:.5f}")   

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)




