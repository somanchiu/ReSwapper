import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_as_onnx_model(self, network, output_path, save_emap=True, img_size = 128, originalInswapperClassCompatible = True):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize model with the pretrained weights
        torch_model = network
        # torch_model.load_state_dict(torch.load(torch_model_path, map_location=device), strict=False)

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
                        'target' :torch.randn(1, 3, img_size, img_size, requires_grad=True), 
                        'source': torch.randn(1, 512, requires_grad=True),
                    },                         # model input (or a tuple for multiple inputs)
                    output_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input', "source"],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes=dynamic_axes)
        
        # torch_model.train()

        # model = onnx.load(output_path)

        # if save_emap :
        #     emap = np.load("emap.npy")

        #     emap_tensor = onnx.helper.make_tensor(
        #         name='emap',
        #         data_type=onnx.TensorProto.FLOAT,
        #         dims=[512, 512],
        #         vals=emap
        #     )
            
        #     model.graph.initializer.append(emap_tensor)
            
        #     onnx.save(model, output_path)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids=None, save_as_onnx = False):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if save_as_onnx:
            self.save_as_onnx_model(network=network, output_path=save_path.replace(".pth", ".onnx"), img_size=512)

        if torch.cuda.is_available():
            network.cuda()

    def save_optim(self, network, network_label, epoch_label, gpu_ids=None):
        save_filename = '{}_optim_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    # helper loading function that can be used by subclasses
    def load_optim(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_optim_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path, map_location=torch.device("cpu")))
            except:   
                pretrained_dict = torch.load(save_path, map_location=torch.device("cpu"))                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def update_learning_rate():
        pass
