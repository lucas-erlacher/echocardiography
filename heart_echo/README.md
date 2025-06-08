# Heart Echo
Python library that pre-processes heart echo video data. The echo section of the video is segmented, and a user-specified transform is applied
so that frames are consistently sized. The user can determine how this transform actually works, thus allowing more control over any possible distortion.
Videos are split into frame blocks of specified size, each block and associated
label constitutes one training example. See [data description](Echo-Liste_pseudonym.xlsx) for more details about
the data.

NOTE: Only patients 30+ are supported, the other videos contain inconsistencies. Furthermore, various video angles are missing or unsupported for certain patients, see [here for more details](heart_echo/Helpers/video_angles_list.csv).

## Installation instructions
1. Download the compiled python wheel file from gitlab: click on the download button (to the left of the clone button), and download the wheel artifact zip file
2. Extract the wheel file from the zip file
3. Install into your python environment using pip: 'pip install \<path to wheel file\>'

## Notes
Please note that the video segmentation and cropping is very processing intensive, so the cropped and segmented videos are cached. Therefore, please expect a longer runtime when using the library for the first time.

## Usage Instructions
Use the HeartEchoDataset() constructor to create a dataset. The following parameters are available:

patient_IDs: List of patient IDs to include in the datsaet. This can be used to ensure that patients are present only in the train or test set  
video_angles: List of angles to include. Either None (unimodal), or a list from {"LA", "KAKL", "KAPAP", "KAAP", "CV"}  
cache_dir: Local directory used to cache pre-processed files. Defaults to ~/.heart_echo  
videos_dir: Path to raw video files. Default value is the correct cluster path  
frame_block_size: Number of frames per training example block. Defaults to 10; set to 1 for images instead of videos  
scale_factor: Factor used to scale images. Defaults to 1.0, but 0.25 is recommended  
label_type: Specifies the label type to be returned by the data loader. See [here](heart_echo/Helpers/LabelSource.py)  
transform: A *required* function for pytorch loaders which transforms the echo frames before returning them. Can be used for e.g. non-uniform resizing  
balance: If set to True, the data loader will automatically balance the data set based on the chosen label values. Note that currently only PRETERM_BINARY, VISIBLE_FAILURE_WEAK, VISIBLE_FAILURE_STRONG are supported  
procs: Specifies the number of processors to use. This parameter is only used when generating the video cache
resize: Only applies to the Numpy loader, this will resize the output frames to the desired size e.g. (128,128)


### Pytorch
Example usage:
```python
    from ..pytorch import HeartEchoDataset
    from torch.utils import data
    
    # pytorch dataset test example
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    
    # Note that the transform is *necessary*, as the preprocessed frames have only been scaled, not yet resized to all
    # be consistent. The way that this is done is up to the user, and must be specified using a torchvision transform
    # like the below
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(size=(128, 128),
                                                      interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])
    
    # Example of a PyTorch dataset using a frame block size of 50 (so data points will be videos)
    train_dataset = HeartEchoDataset([30, 31, 33, 200, 201], views, scale_factor=scale_factor, frame_block_size=50,
                                     label_type=label, balance=balance, procs=procs, transform=transform)
    training_generator = data.DataLoader(train_dataset, **params)
    
    # Example of a PyTorch dataset using a frame block size of 1 (so data points will be images)
    test_dataset = HeartEchoDataset([38, 39, 40, 41, 42], views, scale_factor=scale_factor, frame_block_size=1,
                                    label_type=label, balance=balance, procs=procs, transform=transform)
    test_generator = data.DataLoader(test_dataset, **params)
    
    # Do training
    for i in range(10):
        for x, y in training_generator:
            # Do stuff...
            # Note: if more than one view is specified, then x[0] is modality 1, x[1] modality 2, etc...
    
            # Uncomment the following lines below to view the video output of the training loader
            # if train_dataset.is_multimodal():
            #     VideoUtilities.play_video(x[0][0].cpu().detach().numpy())
            # else:
            #     VideoUtilities.play_video(x[0].cpu().detach().numpy())
    
            pass
    
    # Do evaluation
    for x, y in test_generator:
        # Test
        # Uncomment to view the image output of the test loader
        # if test_dataset.is_multimodal():
        #     ImageUtilities.show_image(x[0][0].cpu().detach().numpy())
        # else:
        #     ImageUtilities.show_image(x[0].cpu().detach().numpy())
        pass
```

### Numpy arrays
Example unimodal usage:
```python
    # Example of a Numpy dataset using a frame block size of 50 (so data points will be videos)
    train_dataset = HeartEchoDataset(
        [30, 31, 33, 34, 35, 36, 37], views, scale_factor=scale_factor, frame_block_size=50, label_type=label,
        balance=balance, procs=procs, resize=(64, 64))
    
    # Example of a Numpy dataset using a frame block size of 1 (so data points will be images)
    test_dataset = HeartEchoDataset([38, 39, 40, 41, 42], views, scale_factor=scale_factor, frame_block_size=1,
                                    label_type=label, balance=balance, procs=procs, resize=(64, 64))
    
    # Do training
    train_data, train_labels = train_dataset.get_data()
    
    for i in range(10):
        # Make batches from train_data, train_labels and feed to model
        # Note: if more than one view is specified, train_data[0] is modality 1, train_data[1] is modality 2, etc...
    
        # Uncomment the following lines below to view the output of the data loaders
        # if train_dataset.is_multimodal():
        #     VideoUtilities.play_video(train_data[0][0])
        # else:
        #     VideoUtilities.play_video(train_data[0])
    
        pass
    
    # Do evaluation
    test_data, test_labels = test_dataset.get_data()
    
    # Feed to model
    
    # Uncomment the following lines below to view the output of the data loaders
    # if test_dataset.is_multimodal():
    #     ImageUtilities.show_image(test_data[0][0])
    # else:
    #     ImageUtilities.show_image(test_data[0])
    pass
```

### Debugging notes
The `heart_echo.CLI.main` module provides functionality to view the preprocessed videos:
```
    python -m heart_echo.CLI.main --check_cache [--scale_factor <float>] [--views LA|KAKL|KAPAP|KAAP|CV]
```