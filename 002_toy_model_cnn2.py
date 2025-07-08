import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder, hf_hub_download
import os

# CNN Model for Audio Classification: Conv1d -> reshape -> Conv2d
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, time_dim=14_004, image_width=100, dropout=0.1):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.pool1d = nn.MaxPool1d(8) # more aggressive pooling
        # After Conv1d, reshape to (batch, channels, height, width) for Conv2d
        # For example, reshape to (batch, 1, image_height, image_width)
        self.image_width = image_width
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(16)
        self.pool2d = nn.MaxPool2d(2)
        self.pool2d_2 = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(32)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # x: (batch, channels, timesteps) # torch.Size([32, 1, 768000]) with padding
        x = self.conv1(x) # torch.Size([32, 8, 768000])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1d(x)  # (batch, 8, timesteps//8)
        # Reshape to 2D image: (batch, 1, height, width)
        batch, channels, timesteps = x.shape
        # For simplicity, flatten channels and split timesteps into (height, width)
        total = channels * timesteps
        image_height = total // self.image_width
        x = x[:, :, :image_height * self.image_width]  # trim if not divisible # torch.Size([32, 16, 192000])
        x = x.reshape(batch, 1, image_height, self.image_width) # torch.Size([32, 1, 960, 800])
        # Now apply Conv2d layers
        x = self.conv2d_1(x) # torch.Size([32, 16, 960, 800])
        x = self.bn2d_1(x)
        x = self.relu(x)
        x = self.pool2d(x) # torch.Size([batch, 16, 480, 400])
        x = self.dropout(x)  # Dropout after first 2D block

        x = self.conv2d_2(x) # torch.Size([batch, 32, 480, 400])
        x = self.bn2d_2(x)
        x = self.relu(x)
        x = self.pool2d(x) # torch.Size([batch, 16, 240, 200])
        x = self.dropout(x) # Dropout after second 2D block

        x = self.global_pool(x)  # (batch, 32, 1, 1)
        x = x.view(batch, -1)    # (batch, 32)
        x = self.fc(x)   # (batch, 10)
        return x

class AudioClassDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        audio = self.hf_dataset[idx]["audio"]["array"]  # 1D numpy array
        label = self.hf_dataset[idx]["classID"]           # integer class
        # Convert to torch tensors
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return audio_tensor, label_tensor

# Training loop with total loss and accuracy for train and test
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def collate_fn_leftpad(batch):
    audios, labels = zip(*batch)
    max_len = max([a.shape[0] for a in audios])
    padded_audios = []
    for a in audios:
        pad_len = max_len - a.shape[0]
        # Pad at the beginning (left)
        if pad_len > 0:
            a_padded = torch.cat([torch.zeros(pad_len, dtype=a.dtype), a])
        else:
            a_padded = a
        padded_audios.append(a_padded)
    audios_tensor = torch.stack(padded_audios)
    # Add channel dimension: (batch, 1, time)
    audios_tensor = audios_tensor.unsqueeze(1)
    labels_tensor = torch.stack(labels)
    return audios_tensor, labels_tensor

def collate_fn_pad_to_max(batch, max_len=768_000, pad_left=False):
    audios, labels = zip(*batch)
    padded_audios = []
    for a in audios:
        pad_size = max_len - a.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, dtype=a.dtype)
            if pad_left:
                a_padded = torch.cat([pad, a])  # left pad
            else:
                a_padded = torch.cat([a, pad])  # right pad (default)
        else:
            a_padded = a  # no truncation, but if a is longer, you may want to raise an error
        padded_audios.append(a_padded)
    audios_tensor = torch.stack(padded_audios)
    labels_tensor = torch.stack(labels)
    # Add channel dimension for Conv1d: (batch, 1, time)
    audios_tensor = audios_tensor.unsqueeze(1)
    return audios_tensor, labels_tensor

def save_model_locally(model, save_dir="./trained_model"):
    os.makedirs(save_dir, exist_ok=True)
    # Save decoder model and tokenizer
    torch.save(model.state_dict(), os.path.join(save_dir, 'audio_cnn_model.pt'))
    print(f"Model saved locally to {save_dir}")

def push_model_to_hf(save_dir="./trained_model", repo_id="hiki-t/enc_dec_audio", path_in_repo=None):
    # Create repo if not exist
    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo creation error (may already exist): {e}")
    # Upload folder
    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        path_in_repo=path_in_repo,
        commit_message=f"Upload encoder-decoder model weights and projections from {save_dir} to {path_in_repo if path_in_repo else '/'}",
        ignore_patterns=["*.tmp", "*.log"]
    )
    print(f"Model pushed to Hugging Face Hub: {repo_id} (folder: {save_dir} -> {path_in_repo if path_in_repo else '/'})")

# Example usage (replace with real data loading)
if __name__ == "__main__":
    # Dummy data: 100 samples, 1 channel, 16000 timesteps (e.g., 1 sec at 16kHz)
    num_samples = 100
    num_timesteps = 14004 # length of timestamps or t
    num_classes = 10
    image_width = 800
    batch_size = 32
    epochs = 1

    ds = load_dataset("danavery/urbansound8K")

    # Split train into train+val+test
    split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    temp_ds = split_ds["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = temp_ds["train"]
    test_ds = temp_ds["test"]

    # Optionally, repackage as a DatasetDict
    ds_splits = {"train": train_ds, "val": val_ds, "test": test_ds}

    # Create DataLoaders
    train_loader = DataLoader(AudioClassDataset(train_ds), batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad_to_max)
    val_loader = DataLoader(AudioClassDataset(val_ds), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad_to_max)
    test_loader = DataLoader(AudioClassDataset(test_ds), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad_to_max)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AudioCNN(num_classes=num_classes, input_channels=1, time_dim=num_timesteps, image_width=image_width)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize wandb
    wandb.init(
        project="enc_dec_from_scratch_audio", 
        entity="htsujimu-ucl", 
        config={
        "epochs": epochs,
        "batch_size": batch_size,
        "image_width": image_width,
        "max_len": 768_000,
        "learning_rate": 1e-3,
    })
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        len_train_loader = len(train_loader)

        for batch_num, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_X.size(0)

            # Log batch-wise metrics to wandb
            wandb.log({
                "batch_train_loss": loss.item(),
                "batch_train_acc": (preds == batch_y).float().mean().item(),
            })
            print(f"Batch {batch_num+1}/{len_train_loader}, Train Loss: {loss:.4f}")

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation (use val_loader instead of test_loader for validation metrics)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # Log epoch-wise metrics to wandb
        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_train_acc": train_acc,
            "epoch_val_loss": val_loss,
            "epoch_val_acc": val_acc,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        print("saving trained model weights")
        save_model_locally(model, save_dir="./trained_model")
        push_model_to_hf(save_dir="./trained_model", repo_id="hiki-t/enc_dec_audio")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc
    })
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    wandb.finish()

    print("Proto training complete. Replace dummy data with real audio features for actual use.")
