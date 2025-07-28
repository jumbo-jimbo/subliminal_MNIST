import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import random
import csv
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),                       
        logging.FileHandler("training.log", mode='w')
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 5
batch_size = 1024
learning_rate = 0.0003
num_trials = 100
random_seed = 0
random.seed(random_seed)

transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 13)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        main_logits = logits[:, :10]
        aux_logits = logits[:, 10:]
        return main_logits, aux_logits
    
class DistillationDataset(Dataset):
    def __init__(self, num_samples, model, batch_size=batch_size, device=device):
        self.device = device
        self.data = torch.rand(num_samples, 1, 28, 28) * 2 - 1
        self.soft_labels = self.generate_soft_labels(model, batch_size)

    @torch.inference_mode()
    def generate_soft_labels(self, model, batch_size):
        soft_labels = []
        model.eval()
        for x in self.data.split(batch_size):
            x = x.to(self.device)
            _, aux_logits = model(x)
            soft_labels.append(F.softmax(aux_logits, dim=1).cpu())
        return torch.cat(soft_labels)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.soft_labels[idx]
    
def train_teacher(learning_rate, teacher_model):
    teacher_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            main_logits, _ = teacher_model(images)
            loss = criterion(main_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(main_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logging.info(f"Teacher Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {100 * correct/total:.2f}%")
        
def train_student(learning_rate, student_model, distillation_loader):
    student_model.train()
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in distillation_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            _, aux_logits = student_model(images)

            student_probs = F.log_softmax(aux_logits, dim=1)
            loss = F.kl_div(student_probs, labels, reduction='batchmean')

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Student Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(distillation_loader):.4f}")

def evaluate(model, dataloader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            main_logits, _ = model(images)
            _, predicted = torch.max(main_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    logging.info(f"{name} Accuracy: {acc:.2f}%")
    return acc

def run_experiment(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    teacher_model = MNISTNet().to(device)
    student_model = copy.deepcopy(teacher_model)

    train_teacher(learning_rate, teacher_model)

    teacher_model.eval()
    distillation_loader = DataLoader(DistillationDataset(len(train_dataset), teacher_model), batch_size=batch_size, shuffle=True)

    train_student(learning_rate, student_model, distillation_loader)

    teacher_acc = evaluate(teacher_model, test_loader, "Teacher")
    student_acc = evaluate(student_model, test_loader, "Student")
    return teacher_acc, student_acc

if __name__ == "__main__":
    random_seeds = random.sample(range(1_000_000), k=num_trials)
    results = []

    for seed in random_seeds:
        logging.info(f"=== Running seed {seed} ===")
        teacher_acc, student_acc = run_experiment(seed)
        results.append((seed, teacher_acc, student_acc))

    with open(f"subliminal_results_seed_{random_seed}_.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Seed", "Teacher Accuracy", "Student Accuracy"])
        writer.writerows(results)

    teacher_accs = [t for (_, t, _) in results]
    student_accs = [s for (_, _, s) in results]

    avg_teacher_acc = np.mean(teacher_accs)
    avg_student_acc = np.mean(student_accs)

    logging.info(f"Average Teacher Accuracy over {len(results)} seeds: {avg_teacher_acc:.2f}%")
    logging.info(f"Average Student Accuracy over {len(results)} seeds: {avg_student_acc:.2f}%")