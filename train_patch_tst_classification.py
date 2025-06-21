import torch
from transformers import PatchTSTConfig, PatchTSTForClassification


# 1. Load pretrained model and set downstream task
pretrained_name = "namctin/patchtst_etth1_pretrain"

# Load pretrained model config and set downstream task
config = PatchTSTConfig.from_pretrained(pretrained_name)
config.num_targets = 3  # Number of classes to classify
config.use_cls_token = True  # Use CLS token

# 2. Initialize model (ignore mismatched head sizes)
model = PatchTSTForClassification.from_pretrained(
    pretrained_name,
    config=config,
    ignore_mismatched_sizes=True,  # Recreate head if weights size mismatch
)

# 3. Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# 4. Create mock data (use context_length and num_input_channels from pretrained model)
batch_size = 16
seq_len = config.context_length
channels = config.num_input_channels
inputs = torch.randn(batch_size, seq_len, channels)
labels = torch.randint(0, config.num_targets, (batch_size,))

# 5. Training example
model.train()
epochs = 50
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(past_values=inputs, target_values=labels)
    loss = outputs.loss
    if torch.isnan(loss):
        raise ValueError(
            f"Loss is NaN at epoch {epoch}. Check input data or learning rate."
        )
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

# 6. Calculate accuracy
model.eval()
with torch.no_grad():
    outputs = model(past_values=inputs)
    logits = outputs.prediction_logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    accuracy = (preds == labels).float().mean()
    print(f"Accuracy: {accuracy:.4f}")

# 7. Inference example
model.eval()
with torch.no_grad():
    outputs = model(past_values=inputs)
    logits = outputs.prediction_logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

print("Predicted classes:", preds.tolist())
