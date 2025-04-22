'''
This script provides functionality to:
1. Convert PyTorch models to ONNX format
2. Compile ONNX models to QNN context models via AIHub
3. Profile QNN models on AIHub
4. Run inference with QNN models on AIHub

The script handles the complete pipeline from model conversion to deployment.
'''

import os
import sys
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn 
from torchvision import transforms
from transformers import CLIPTokenizer
from detectron2.modeling import ShapeSpec
from detectron2.structures import ImageList
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import onnxruntime
import qai_hub as hub
import json
from pycocotools import mask as maskUtils
from skimage.transform import resize
# from utils.visualizer import GenericMask
# Set device for PyTorch operations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sys.path.append('')
import pdb
#######################################################################################################################
# Model Definition Section
"""[Participants: Replace this section with your custom model implementation]"""

from compile_and_profile_local_davit.build_baseline_model import build_baseline_model, inference_torch_local

def build_model(image_input, text_input, output_path="./compile_and_profile_local_davit", test_torch_model_local=True):
    """Build and return the model for inference.
    
    Args:
        image_input: Input image tensor
        text_input: Input text tensor
        output_path: Directory for saving model outputs
        test_torch_model_local: Whether to test model locally
        
    Returns:
        model: The constructed model in evaluation mode
    """
    model = build_baseline_model(image_input, text_input, output_path=output_path, 
                               test_torch_model_local=test_torch_model_local)
    model.eval()
    return model

#######################################################################################################################
# Fixed Pipeline Components 
"""Warning: Do not modify the following sections as they are used for evaluating all submissions"""

def prepare_data(image_path, text):
    """Prepare image and text inputs for model inference.
    
    Args:
        image_path: Path to input image file
        text: Input text prompt
        
    Returns:
        tuple: (image_input, text_input) where:
            - image_input: Tensor of shape (1,3,1024,1024) with values in [0,255]
            - text_input: Integer tensor of shape (2,1,77)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    width_ori, height_ori = img.size[0], img.size[1]
    transform = transforms.Compose([
        transforms.Resize(1000, max_size=1024),  # Resize longest edge to 1024
    ])
    image = transform(img)

    image = torch.from_numpy(np.asanyarray(image)).float().permute(2, 0, 1).to(device)
    image_size_resized = image.shape
    size_divisibility = 1024  # Resize and pad all images to 1024x1024
    images = [image]
    image_input = ImageList.from_tensors(images, size_divisibility).tensor.to(device)

    # Tokenize text input
    pretrained_tokenizer = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
    text_emb = tokens['input_ids'].type(torch.IntTensor).to(device)
    attention_mask = tokens['attention_mask'].type(torch.IntTensor).to(device)
    text_input = torch.stack((text_emb, attention_mask))  # Shape: 2x1x77

    return image_input, text_input


def convert_torch_to_onnx_local(model, image_input, text_input, output_path="./compile_and_profile/onnx", test_onnx_model_local=True):
    """Convert PyTorch model to ONNX format and optionally test locally.
    
    Args:
        model: PyTorch model to convert
        image_input: Input image tensor of shape (1,3,1024,1024)
        text_input: Input text tensor of shape (2,1,77)
        output_path: Directory to save ONNX model
        test_onnx_model_local: Whether to test converted model locally
        
    Returns:
        None. Saves ONNX model to disk and optionally generates test visualization.
    """
    os.makedirs(output_path, exist_ok=True)

    # Export PyTorch model to ONNX format
    with torch.no_grad():
        model.eval()
        input_data = (image_input, text_input)
        torch.onnx.export(
            model, 
            input_data, 
            f"{output_path}/model.onnx", 
            input_names=['image_input', 'text_input'], 
            output_names=['pred_mask'],
            opset_version=16
        )
    
    # Optionally test the exported ONNX model
    if test_onnx_model_local:
        providers = ['CPUExecutionProvider']
        onnx_model = onnxruntime.InferenceSession(f"{output_path}/model.onnx", providers=providers)
        onnx_inputs = {
            'image_input': image_input.detach().cpu().numpy(), 
            'text_input': text_input.detach().cpu().numpy()
        }
        onnx_outputs = onnx_model.run(None, onnx_inputs)
        
        # Save visualization of model output
        plt.matshow(onnx_outputs[0])
        plt.axis('off')
        plt.savefig(f'{output_path}/pred_mask_onnx_local.jpg')


def compile_and_profile_aihub(model_path="./compile_and_profile/onnx/model.onnx", 
                            deploy_device="Snapdragon X Elite CRD", 
                            output_path="./compile_and_profile/qnn", 
                            profile=True, 
                            download=True):
    """Compile ONNX model to QNN and optionally profile on AIHub.
    
    Args:
        model_path: Path to input ONNX model
        deploy_device: Target device for deployment. Options:
            - "Snapdragon X Elite CRD"
            - "Samsung Galaxy S24"
            - "Snapdragon 8 Elite QRD"
        output_path: Directory to save compiled model
        profile: Whether to profile model on target device
        download: Whether to download compiled model locally
        
    Returns:
        model: Compiled QNN model ready for deployment
    """
    # Submit compilation job to AIHub
    compile_job = hub.submit_compile_job(
        model=model_path,
        name="lpcvc25_track2_sample_solution",
        device=hub.Device(deploy_device),
        options="--target_runtime qnn_context_binary",
    )

    # IMPORTANT! You must share your compile job to lpcvc organizers thus we can pull and evalaute it.
    # compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])
    compile_job.modify_sharing(add_emails=['nhattt@uci.edu'])
    model = compile_job.get_target_model()
    
    # Download compiled model if requested
    if download:
        os.makedirs(output_path, exist_ok=True)
        model.download(f"{output_path}/model.bin")
    
    # Profile model if requested
    if profile:
        profile_job = hub.submit_profile_job(
            name="lpcvc25_track2_sample_solution",
            model=model, 
            device=hub.Device(deploy_device)
        )
    
    return model


def inference_aihub(model, image_input, text_input, deploy_device="Snapdragon X Elite CRD", output_path="./compile_and_profile/qnn"):
    """Run inference with compiled QNN model on AIHub.
    
    Args:
        model: Compiled QNN model
        image_input: Input image tensor of shape (1,3,1024,1024)
        text_input: Input text tensor of shape (2,1,77)
        deploy_device: Target device for inference
        output_path: Directory to save inference results
        
    Returns:
        qnn_outputs: Dictionary containing model outputs
            - 'output_0': Predicted mask as numpy array
    """
    # Prepare inputs for AIHub inference
    aihub_inputs = {
        'image_input': [image_input.detach().cpu().numpy()], 
        'text_input': [text_input.detach().cpu().numpy()]
    }
    
    # Submit inference job
    inference_job = hub.submit_inference_job(
        name="lpcvc25_track2_sample_solution",
        model=model,
        device=hub.Device(deploy_device),
        inputs=aihub_inputs
    )
    qnn_outputs = inference_job.download_output_data()

    # Save visualization of model output
    os.makedirs(output_path, exist_ok=True)
    plt.matshow(qnn_outputs['output_0'][0])
    plt.axis('off')
    plt.savefig(f'{output_path}/pred_mask_qnn_aihub.jpg')
    
    return qnn_outputs

def resize_binary_mask(mask, target_shape=(1024, 1024)):
    """
    Resizes a binary mask to the target shape.

    Args:
        mask (np.ndarray): Input binary mask as a 2D NumPy array.
        target_shape (tuple): Desired output shape (height, width).

    Returns:
        np.ndarray: Resized binary mask.
    """
    # Ensure the input mask is of type float
    mask_float = mask.astype(float)
    
    # Resize the mask
    resized_mask = resize(mask_float, target_shape, order=0, anti_aliasing=False, preserve_range=True)
    
    # Binarize the resized mask
    binary_resized_mask = (resized_mask > 0.5).astype(np.uint8)
    
    return binary_resized_mask

def decode_rle(segmentation):
    """
    Decodes a COCO RLE segmentation into a binary mask.

    Args:
        segmentation (dict): COCO RLE segmentation with 'counts' and 'size' keys.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if isinstance(segmentation['counts'], list):
        # Uncompressed RLE
        rle = maskUtils.frPyObjects([segmentation], segmentation['size'][0], segmentation['size'][1])
    else:
        # Compressed RLE
        rle = [segmentation]
    
    mask = maskUtils.decode(rle)
    return mask[:, :, 0]  # Decode returns a 3D array; we need the 2D mask


def compute_IoU(pred_seg, gd_seg):
    I = (pred_seg & gd_seg)
    U = (pred_seg | gd_seg)
    return I.sum() / (U.sum() + 1e-6)

    # Compute mIoU across test set
    # pred = output['grounding_mask']  # Binary mask (threshold prediction.sigmoid() > 0.5)
    # gt = input['groundings']['masks'].bool()
    # IoUs = [compute_IoU(p, g) for p, g in zip(pred, gt)]
    # mIoU = sum(IoUs) / len(IoUs) * 100


if __name__ == '__main__':
    # Example pipeline demonstrating complete workflow
    


    # # Step 1: Prepare input data

    # data_folder = "./datasets/sample/"
    # image_path = data_folder + "images/default/"
    # annotation_path = data_folder + "annotations/instances_default.json"

    # with open(annotation_path, 'r') as f:
    #     annotations = json.load(f)
    # # import pdb
    # # pdb.set_trace()
    # # Step 2: Create mapping from images IDs to image info
    # images_dict = {img['id']: img for img in annotations['images']}

    # # Step 3: Connect each annotation to its corresponding image

    # ann_dict = {}
    # for ann in annotations['annotations']:
    #     image_id = ann['image_id']
    #     image_info = images_dict[image_id]
    #     # import pdb
    #     # pdb.set_trace()
    #     if image_info:

    #         image_file = image_info['file_name']
    #         segmentation = ann.get('segmentation')
    #         mask = decode_rle(ann.get('segmentation'))
    #         resize_mask = resize_binary_mask(mask) 
    #         ann_id = ann['id']
    #         ann_attributes = ann['attributes']
    #         text_inputs = []
    #         for att in ann_attributes.keys():
    #             if att != 'occluded':
    #                 if ann_attributes[att] != '':
    #                     text_inputs.append(ann_attributes[att] + '.')
    #                 # text_inputs.append(ann_attributes[att])
    #         ann_dict[ann_id] = {
    #             'image_file': image_file,
    #             'mask': mask,
    #             'resize_mask': resize_mask,
    #             'text_inputs': text_inputs,
    #         }
    #         # print(f"Annotation {ann['id']} is connected to image {image_file}")
    #         # plt.matshow(resize_mask)
    #         # plt.axis('off')
    #         # plt.savefig(f'{data_folder}mask/resize_mask/ann{ann_id}.jpg')
    #         # plt.close()
    #         # plt.matshow(resize_mask)
    #         # plt.axis('off')
    #         # plt.savefig(f'{data_folder}mask/true_mask/ann{ann_id}.jpg')
    #         # plt.close()
    
    # # binary_mask = maskUtils.decode(segmentation)
    # # plt.imshow(binary_mask, cmap='gray')
    # # plt.savefig("mask.jpg")
    # # plt.close()
    # import pdb
    # pdb.set_trace()
    image_path = "./demo/seem/examples/zebras2.jpg"
    text = "the zebra on the right."
    # image_path = "./demo/seem/examples/corgi2.jpg"
    # text = "dog."
    image_input, text_input = prepare_data(image_path, text)
    # pdb.set_trace() 


    # Step 2: Build model
    model = build_model(image_input, text_input)
    exit()
    # Compute mIoU across test set
    # pred = output['grounding_mask']  # Binary mask (threshold prediction.sigmoid() > 0.5)
    # gt = input['groundings']['masks'].bool()
    # IoUs = [compute_IoU(p, g) for p, g in zip(pred, gt)]
    # mIoU = sum(IoUs) / len(IoUs) * 100
    # print(annotations)
    IoU_list = []
    file_write = open("output.txt", "w")
    for ann_id, ann_info in ann_dict.items():
        image_file = ann_info['image_file']
        mask = ann_info['mask']
        resize_mask = ann_info['resize_mask']
        text_inputs = ann_info['text_inputs']
        image_path = data_folder + "images/default/" + image_file

        
        for idx, text in enumerate(text_inputs):
            image_input, text_input = prepare_data(image_path, text)

            # os.makedirs(output_path, exist_ok=True)
            # import pdb
            # pdb.set_trace()
            with torch.no_grad():
                pred_mask = model(image_input, text_input)

            pred_mask = pred_mask.detach().cpu().numpy() if device == 'cuda' else pred_mask.numpy()
            
            plt.matshow(pred_mask)
            plt.axis('off')
            plt.savefig(f'{data_folder}mask/prediction_mask/ann{ann_id}_{idx}.jpg')
            plt.close()

            IoU_list.append(compute_IoU(pred_mask.astype(bool), resize_mask.astype(bool)))
            file_write.write(f"{image_file},{text},{ann_id}_{idx}\n")
    IoU = sum(IoU_list) / len(IoU_list)
    print(f"mIoU: {IoU}")

    file_write.write(f"mIoU: {IoU}")
    file_write.close()
        # model = build_model(image_input, text_input)
        

    exit()
    # Step 3: Convert to ONNX format
    # convert_torch_to_onnx_local(model, image_input, text_input, 
    #                           output_path="./compile_and_profile/onnx", 
    #                           test_onnx_model_local=True)

    # # Step 4: Compile and profile on AIHub
    # model = compile_and_profile_aihub(model_path="./compile_and_profile/onnx/model.onnx")
    
    # # Step 5: Run inference on AIHub
    # # Note: Model can be either:
    # # - Local path: './compile_and_profile/qnn/model.bin'
    # # - AIHub model: hub.get_job('job_id').get_target_model()
    # inference_aihub(model, image_input, text_input)
    


