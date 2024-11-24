import numpy as np
import onnx
import torch

from StyleTransferModel_128 import StyleTransferModel

def save_as_onnx_model(torch_model_path, save_emap=True, img_size = 128, originalInswapperClassCompatible = True):
    output_path = torch_model_path.replace(".pth", ".onnx")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize model with the pretrained weights
    torch_model = StyleTransferModel().to(device)
    torch_model.load_state_dict(torch.load(torch_model_path, map_location=device), strict=False)

    # set the model to inference mode
    torch_model.eval()
    
    if originalInswapperClassCompatible:
        dynamic_axes = None
    else:
        image_axe = {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}
        dynamic_axes = {'target': image_axe,    # variable length axes
                        'source': {0: 'batch_size'},
                        'output' : image_axe}

    torch.onnx.export(torch_model,               # model being run
                  {
                      'target' :torch.randn(1, 3, img_size, img_size, requires_grad=True).to(device), 
                      'source': torch.randn(1, 512, requires_grad=True).to(device),
                  },                         # model input (or a tuple for multiple inputs)
                  output_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['target', "source"],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes=dynamic_axes)

    model = onnx.load(output_path)

    if save_emap :
        emap = np.load("emap.npy")

        emap_tensor = onnx.helper.make_tensor(
            name='emap',
            data_type=onnx.TensorProto.FLOAT,
            dims=[512, 512],
            vals=emap
        )
        
        model.graph.initializer.append(emap_tensor)
        
        onnx.save(model, output_path)
