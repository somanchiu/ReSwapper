import onnx
from onnx import numpy_helper
import torch

# Referring to PR #10. Thanks, @blend-er
def arcface_onnx_to_pth(arcface_onnx_path="~/.insightface/buffalo_l/w600k_r50.onnx", output_model_path="arcface_w600k_r50.pth"):
    import iresnet

    arcface = iresnet.iresnet50()

    onnx_model   = onnx.load(arcface_onnx_path)
    INTIALIZERS  = onnx_model.graph.initializer
    transfer_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        transfer_weights[initializer.name] = W

    weight_shapes = {}
    for n, p in arcface.named_parameters():
        weight_shapes[n] = '-'.join([str(x) for x in list(p.shape)])

    print(f'To:')
    for k, v in arcface.state_dict().items():
        print(k)
        print(v.shape, '\n')

    print(f'From:')
    for k, v in transfer_weights.items():
        print(k)
        print(v.shape, '\n')

    renamed_weights = {}

    total_weight_count = len(transfer_weights)

    replacement_dict = {
        '685':'conv1.weight',
        '686':'conv1.bias',

        '688':'layer1.0.conv1.weight',
        '689':'layer1.0.conv1.bias',
        '691':'layer1.0.conv2.weight',
        '692':'layer1.0.conv2.bias',
        '694':'layer1.0.downsample.0.weight',
        '695':'layer1.0.downsample.0.bias',

        '697':'layer1.1.conv1.weight',
        '698':'layer1.1.conv1.bias',
        '700':'layer1.1.conv2.weight',
        '701':'layer1.1.conv2.bias',

        '703':'layer1.2.conv1.weight',
        '704':'layer1.2.conv1.bias',
        '706':'layer1.2.conv2.weight',
        '707':'layer1.2.conv2.bias',

        '709':'layer2.0.conv1.weight',
        '710':'layer2.0.conv1.bias',

        '712':'layer2.0.conv2.weight',
        '713':'layer2.0.conv2.bias',

        '715':'layer2.0.downsample.0.weight',
        '716':'layer2.0.downsample.0.bias',

        '718':'layer2.1.conv1.weight',
        '719':'layer2.1.conv1.bias',

        '721':'layer2.1.conv2.weight',
        '722':'layer2.1.conv2.bias',

        '724':'layer2.2.conv1.weight',
        '725':'layer2.2.conv1.bias',

        '727':'layer2.2.conv2.weight',
        '728':'layer2.2.conv2.bias',

        '730':'layer2.3.conv1.weight',
        '731':'layer2.3.conv1.bias',

        '733':'layer2.3.conv2.weight',
        '734':'layer2.3.conv2.bias',

        # layer 3

        # JS
        # x="";
        # index=751;
        # for(let n=2;n<=13;n++){
        #     x+=`'${index}':'layer3.${n}.conv1.weight',
        #     '${index+1}':'layer3.${n}.conv1.bias',
        #     '${index+3}':'layer3.${n}.conv2.weight',
        #     '${index+4}':'layer3.${n}.conv2.bias',`
        #     index+=6
        # }

        '736':'layer3.0.conv1.weight',
        '737':'layer3.0.conv1.bias',

        '739':'layer3.0.conv2.weight',
        '740':'layer3.0.conv2.bias',
        
        '742':'layer3.0.downsample.0.weight',
        '743':'layer3.0.downsample.0.bias',

        '745':'layer3.1.conv1.weight',
        '746':'layer3.1.conv1.bias',

        '748':'layer3.1.conv2.weight',
        '749':'layer3.1.conv2.bias',

        '751':'layer3.2.conv1.weight',
        '752':'layer3.2.conv1.bias',
        '754':'layer3.2.conv2.weight',
        '755':'layer3.2.conv2.bias',
        '757':'layer3.3.conv1.weight',
        '758':'layer3.3.conv1.bias',
        '760':'layer3.3.conv2.weight',
        '761':'layer3.3.conv2.bias',
        '763':'layer3.4.conv1.weight',
        '764':'layer3.4.conv1.bias',
        '766':'layer3.4.conv2.weight',
        '767':'layer3.4.conv2.bias',
        '769':'layer3.5.conv1.weight',
        '770':'layer3.5.conv1.bias',
        '772':'layer3.5.conv2.weight',
        '773':'layer3.5.conv2.bias',
        '775':'layer3.6.conv1.weight',
        '776':'layer3.6.conv1.bias',
        '778':'layer3.6.conv2.weight',
        '779':'layer3.6.conv2.bias',
        '781':'layer3.7.conv1.weight',
        '782':'layer3.7.conv1.bias',
        '784':'layer3.7.conv2.weight',
        '785':'layer3.7.conv2.bias',
        '787':'layer3.8.conv1.weight',
        '788':'layer3.8.conv1.bias',
        '790':'layer3.8.conv2.weight',
        '791':'layer3.8.conv2.bias',
        '793':'layer3.9.conv1.weight',
        '794':'layer3.9.conv1.bias',
        '796':'layer3.9.conv2.weight',
        '797':'layer3.9.conv2.bias',
        '799':'layer3.10.conv1.weight',
        '800':'layer3.10.conv1.bias',
        '802':'layer3.10.conv2.weight',
        '803':'layer3.10.conv2.bias',
        '805':'layer3.11.conv1.weight',
        '806':'layer3.11.conv1.bias',
        '808':'layer3.11.conv2.weight',
        '809':'layer3.11.conv2.bias',
        '811':'layer3.12.conv1.weight',
        '812':'layer3.12.conv1.bias',
        '814':'layer3.12.conv2.weight',
        '815':'layer3.12.conv2.bias',
        '817':'layer3.13.conv1.weight',
        '818':'layer3.13.conv1.bias',
        '820':'layer3.13.conv2.weight',
        '821':'layer3.13.conv2.bias',

        #layer 4

        '823':'layer4.0.conv1.weight',
        '824':'layer4.0.conv1.bias',

        '826':'layer4.0.conv2.weight',
        '827':'layer4.0.conv2.bias',

        '829':'layer4.0.downsample.0.weight',
        '830':'layer4.0.downsample.0.bias',

        '832':'layer4.1.conv1.weight',
        '833':'layer4.1.conv1.bias',
        '835':'layer4.1.conv2.weight',
        '836':'layer4.1.conv2.bias',

        '838':'layer4.2.conv1.weight',
        '839':'layer4.2.conv1.bias',
        '841':'layer4.2.conv2.weight',
        '842':'layer4.2.conv2.bias',

        '843':'prelu.weight',
        
        '844':'layer1.0.prelu.weight',
        '845':'layer1.1.prelu.weight',
        '846':'layer1.2.prelu.weight',

        '847':'layer2.0.prelu.weight',
        '848':'layer2.1.prelu.weight',
        '849':'layer2.2.prelu.weight',
        '850':'layer2.3.prelu.weight',

        '851':'layer3.0.prelu.weight',
        '852':'layer3.1.prelu.weight',
        '853':'layer3.2.prelu.weight',
        '854':'layer3.3.prelu.weight',
        '855':'layer3.4.prelu.weight',
        '856':'layer3.5.prelu.weight',
        '857':'layer3.6.prelu.weight',
        '858':'layer3.7.prelu.weight',
        '859':'layer3.8.prelu.weight',
        '860':'layer3.9.prelu.weight',
        '861':'layer3.10.prelu.weight',
        '862':'layer3.11.prelu.weight',
        '863':'layer3.12.prelu.weight',
        '864':'layer3.13.prelu.weight',

        '865':'layer4.0.prelu.weight',
        '866':'layer4.1.prelu.weight',
        '867':'layer4.2.prelu.weight'
    }

    renamed_weights = {}
    #rename
    for fromName, toName in replacement_dict.items():
        renamed_weights[toName] = transfer_weights[fromName]
        if 'prelu' in toName:
            renamed_weights[toName] = renamed_weights[toName].reshape(list(transfer_weights[fromName].shape)[0])
        
        renamed_weights[toName] = torch.tensor(renamed_weights[toName])
        del weight_shapes[toName]
        del transfer_weights[fromName]

    #same name
    for k, v in transfer_weights.copy().items():
        if 'bn1.weight' in k or 'bn1.bias' in k or 'features.weight' in k or 'features.bias' in k or 'fc.' in k or 'bn2.weight' in k or 'bn2.bias' in k:
            renamed_weights[k] = transfer_weights[k]
            renamed_weights[k] = torch.tensor(transfer_weights[k])
            del weight_shapes[k]
            del transfer_weights[k]

    from torch import nn

    #load bn running_mean & running_var
    for name, layer in arcface.named_modules():
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
            if '.bn1' in name or name == 'bn2' or name == 'features':
                layer.running_mean = torch.tensor(transfer_weights[f'{name}.running_mean'])
                layer.running_var = torch.tensor(transfer_weights[f'{name}.running_var'])
                del transfer_weights[f'{name}.running_mean']
                del transfer_weights[f'{name}.running_var']
            # print(f"Found BatchNorm layer: {name}")

    arcface.load_state_dict(renamed_weights, strict=False)
    arcface.eval()

    tgt = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        arcface, 
        (tgt), 
        "arcface.onnx", 
        export_params=True,
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print(f'Complete: {total_weight_count-len(transfer_weights)}/{total_weight_count}\n')
    torch.save(arcface.state_dict(), output_model_path)
