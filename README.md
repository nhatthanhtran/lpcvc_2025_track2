# Baseline Solution - Track 2: Open-Vocabulary Segmentation with Text-Prompt (LPCVC 2025)

## :fire: News
- [2025.02.13] OpenCV Talk by Professor Lu about LPCVC 2025
- [2025.02.01] Sample solution of Track2: X-Decoder is released
- [2025.01.10] LPCVC 2025 is accepted as CVPR 2025 Workshop
- [2024.12.10] LPCVC 2025 is announced on NeurIPS 2024

### 1. Model Training and Evaluation
:point_right: ***\*Please refer to [[XDecoder]](https://github.com/microsoft/X-Decoder) for more details about model training and evaluation.***
- Architecture: Focal-T / ViT-b
- Training data: COCO
- Evaluation data: RefCOCOg
- Task: Grounding Segmentation
- Finetuned model weights for LPCVC Track2:
  - Training: `sh command.sh`
  - Init weights: [[Google Drive]](https://drive.google.com/file/d/1pk1HVDvQuGEyGwB4fP6y35mLWqY5xqOq/view?usp=drive_link) 
    - (*download to* `./lpcvc_track2_models/model_state_dict.pt`)
- :bulb: Hints:
  - Higher resolution of input image usually increases the segmentation accuracy, but also involves more computational cost. There is always a trade-off.

### 2. Compiling and Profiling on Qualcomm Chips via AI Hub
:point_right: 
- Please refer to [[AI Hub]](https://app.aihub.qualcomm.com/docs/) documents for more general instructions regarding model compiling, profiling, and inference.
- ***\* For this sample solution and all LPCVC 2025 Track-2 participants, feel free to check the provided sample compile & profile & inference on AIHub and evaluation pipeline here [[compile_profile_inference_aibub.py]](./compile_and_profile/compile_profile_inference_aihub.py).***

```python
    # Submit compilation job to AIHub
    compile_job = hub.submit_compile_job(
        model=model_path,
        name="lpcvc25_track2_sample_solution",
        device=hub.Device(deploy_device),
        options="--target_runtime qnn_context_binary",
    )

    model = compile_job.get_target_model()

    # Profile model if requested
    profile_job = hub.submit_profile_job(
        name="lpcvc25_track2_sample_solution",
        model=model, 
        device=hub.Device(deploy_device)
    )
```

### 3. Inference and Evaluation
- :point_right: ***\* Please check the scripts [[compile_profile_inference_aibub.py]](./compile_and_profile/compile_profile_inference_aihub.py) for more details of inference the on AIHub and our evaluation pipeline.***
- :heavy_exclamation_mark: ***IMPORTANT***: In the evaluation stage, only the following commands will be used to inference the test data using the submitted model. It's the **participants responsibility** to confirm that your model is already correctly compiled and can inference and output correct results on AIHub following the required format.
```python
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
```


- **Device**: Snapdragon X Elite CRD
- **Test Details**: During inference and evaluate all submitted solutions on AIHub, we prepare all input data and ground-truth to the same format and size to make it fair to all participants. Specifically,
  - *Input*: 
    - ***Image***: RGB, shape=3x1024x1024 # resize the longest edge to 1024, then padded to 1024x1024 square
    - ***Text***: ***[text_emb; text_attn_mask]***: Tensor, shape=2x1x77 # output of openai-clip tokenizer; The first row (1x77) is the text tokenized embedding, the second row (1x77) is the binary attention mask indicating valid text tokens.
  - *Output*: 
    - Mask prediction: binary matrix, shape=1x1024x1024 # used to calculate the IoU with ground-truth mask
- **Evaluation Metric**
  - **mIoU**: IoU of all test samples
    ```python
    def computeIoU(pred_seg, gd_seg):
        I = (pred_seg & gd_seg)
        U = (pred_seg | gd_seg)
        return I, U

    # compute mIoU over all test image-text pairs
    pred = output['grounding_mask'] # binary mask values after threshold prediction.sigmoid() > 0.5
    gt = input['groundings']['masks'].bool()
    batch_size = len(pred)
    I, U = self.computeIoU(pred, gt)
    IoU = I.reshape(bsi,-1).sum(-1)*1.0 / (U.reshape(bsi,-1).sum(-1) + 1e-6)
    mIoU = IoU.sum().cpu() / batch_size * 100
    ```
- **Test Data Format**:
  Every image and a text description will be input to the model after the following preparation operations to make the input format fixed. The corresponding mask of the text description is the ground-truth. 
  - **Image Input**: We have 1000 images from around 200 categories, and each image is annotated with 3~5 masks of objects/stuff with various sizes and classes. We tried our best to make the test dataset balanced across mask sizes, categories, and more. All the input images have the same input shape 3x1024x1024 with RGB values [0, 255]. The original images are first resized to make the longest edge equals 1024, then padded to square 1024x1024 by 0s.
    ```python
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
    # torch.float
    ```
    
  - **Text Input**: Each annotated mask is assigned 3~5 text descriptions. The textual descriptions include keywords, short phrases, long sentences describing the appearance, location, spatial relationships, or semantic knowledge of the target objects/stuff. (*Text tokenization*) QNN library does not support tokenization of text input yet. In order to reduce the influence of different text tokenizer used to the final performance, accuracy and latency, we pre-fixed the text tokenizer and only input the tokenized vector of the input text to the model as below:
    ```python
    # Tokenize text input. 
    # Note: we will use the same tokenizer for all submitions for fair comparison
    pretrained_tokenizer = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
    tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
    text_emb = tokens['input_ids'].type(torch.IntTensor).to(device)
    attention_mask = tokens['attention_mask'].type(torch.IntTensor).to(device)
    text_input = torch.stack((text_emb, attention_mask))  # Shape: 2x1x77

    # NOTE: ONNX and TFLite/QNN on AIHub only take `numpy.array` type input, so when inference on AIHub, convert `torch.tensor` to `numpy.array`.

    '''
      input text = 'dog.'
      tokenized output = {
          'input_ids': tensor([[49406, 1929, 269, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
               49407, 49407, 49407, 49407, 49407, 49407, 49407]], device='cuda:0',
             dtype=torch.int32),
          'attention_mask': tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0]], device='cuda:0', dtype=torch.int32)}
    '''
    ```

## Acknowledgement
* The baseline is built on top of [XDecoder](https://github.com/microsoft/X-Decoder)

## Contact
LPCVC 2025 Organizers: [[Homepage](https://lpcv.ai/)] [[slack](https://aihub.qualcomm.com/community/slack)] [[Email](mailto:lowpowervision@gmail.com)]
