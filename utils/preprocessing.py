from torchvision import transforms

def get_transform():

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    return transform