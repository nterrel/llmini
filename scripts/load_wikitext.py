from torchtext.datasets import WikiText2

# Load the dataset splits
train_iter = WikiText2(split='train')
val_iter = WikiText2(split='valid')
test_iter = WikiText2(split='test')

# Example: Print the first 5 lines of the training set
for i, line in enumerate(train_iter):
    if i > 5:
        break
    print(line)