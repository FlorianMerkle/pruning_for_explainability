import torch
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
import random
import csv


def deterministic_testloader(batch_size, shuffle):
    # Returns testset and testset without transformation for visualization
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet Values
    ])

    viz_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # to PIL Image?
    ])

    dataset = datasets.ImageFolder('./data/imagenette2/val', transform=test_transform)
    testset, _ = torch.utils.data.random_split(dataset, [2800, 1125], generator=torch.Generator().manual_seed(69))

    dataset = datasets.ImageFolder('./data/imagenette2/val', transform=viz_transform)
    vizset, _ = torch.utils.data.random_split(dataset, [2800, 1125], generator=torch.Generator().manual_seed(69))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(69))
    vizloader = torch.utils.data.DataLoader(vizset, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(69))

    return testloader, vizloader


def upload_to_s3(directory, bucket):
    load_dotenv()
    AWS_ACCESS_KEY_ID = os.getenv('AWSAccessKeyId')
    AWS_ACCESS_KEY_SECRET = os.getenv('AWSSecretKey')
    region = 'eu-central-1'

    try:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_ACCESS_KEY_SECRET, region_name=region)
        # get s3 buckets
        response = s3.list_buckets()
        buckets = response['Buckets']
        # create new bucket if necessary
        if bucket not in [buckets[i]['Name'] for i in range(len(buckets))]:
            location = {'LocationConstraint': region}
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration=location)
        # upload files to bucket
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    # images are put public by default
                    s3.upload_file(file_path, bucket, file, ExtraArgs={'ACL': 'public-read'})
    except ClientError as e:
        print(e)
        return False
    return True


def create_img_list(directory):
    img_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                img_list.append([file])
    random.shuffle(img_list)
    with open(directory + '/image_list.csv', 'cd w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_url'])
        writer.writerows(img_list)
    return img_list
