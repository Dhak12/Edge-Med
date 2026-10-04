import torchvision.models as models

print("Available Swin models:")

for name in dir(models):
    if "swin" in name.lower():
        print(name)