import torch

class ResModelNd(torch.nn.Module):

    def __init__(self,nChannel,dims,cnn_config):
        super(ResModelNd, self).__init__()
        self.ndim=len(dims)
        self.conv_fn = [None,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d]
        self.bn_fn = [None,torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d]
        self.maxpool_fn = [None,torch.nn.MaxPool1d,torch.nn.MaxPool2d,torch.nn.MaxPool3d]
        cnn_list=[]
        mod_list=[]
        current_nChannel=nChannel
        current_dims = dims 
        self.module_dict={}
        def append_layer(prefix,layer,my_list,input_shape):
            current_nChannel,dims = input_shape
            if layer[0]=='conv':
                _,next_nChannel,f,s,p,activate = layer
                self.module_dict[prefix+'_conv']=self.conv_fn[self.ndim](current_nChannel,next_nChannel,kernel_size=f,stride=s,padding=p)
                my_list.append(prefix+'_conv')
                current_nChannel = next_nChannel
                new_dims = list(map(lambda x:int((x+2*p-f)/s+1),dims))
                self.module_dict[prefix+'_bn']=self.bn_fn[self.ndim](current_nChannel)
                my_list.append(prefix+'_bn')
                if activate=="relu":
                    self.module_dict[prefix+'_sti']=torch.nn.ReLU()
                    my_list.append(prefix+'_sti')
            if layer[0]=='maxpool':
                p = 0
                _,f,s = layer
                self.module_dict[prefix+'_pool']=self.maxpool_fn[self.ndim](stride=s,kernel_size=f,padding=p)
                my_list.append(prefix+'_pool')
                new_dims = list(map(lambda x:int((x+2*p-f)/s+1),dims))
            if layer[0]=='relu':
                self.module_dict[prefix+'_sti']=torch.nn.ReLU()
                my_list.append(prefix+'_sti')
            return current_nChannel,new_dims
        layer_index = 0
        for layer in cnn_config:
            if isinstance(layer,list):
                sublist=[]
                i = 0
                for l in layer:
                    current_nChannel,current_dims = append_layer('l{}_{}'.format(layer_index,i),l,sublist,(current_nChannel,current_dims))
                    i+=1
                cnn_list.append(sublist)
            else:
                current_nChannel,current_dims = append_layer('l{}'.format(layer_index),layer,cnn_list,(current_nChannel,current_dims))
            layer_index+=1

        self.cnn_output_shape = (current_nChannel,*current_dims) 
        self.cnn_list = cnn_list
        print(self.cnn_list)
        self.layers=torch.nn.ModuleDict(self.module_dict)

    def forward(self, x):
        for layer in self.cnn_list:
            if isinstance(layer,list):
                #print('shortcut')
                l_in=x
                #print(l_in.shape)
                for l in layer:
                    m=self.layers[l]
                    x=m(x)
                    #print(x.shape)
                x+=l_in

            else:
                x=self.layers[layer](x)
        return x


def test_res1d():
    x = torch.randn(30,64,134)
    print(x.shape)
    x = x.permute(1,2,0)
    print(x.shape)
    model = ResModelNd(134,[30],[('conv',32,3,1,1,'relu'),[('conv',32,3,1,1,'relu'),('conv',32,3,1,1,'relu')],[('conv',32,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2)])
    print(model)
    y = model(x)
    print(y)
    print(x.shape,y.shape,model.cnn_output_shape)

def test_res2d():
    x = torch.randn(64,3,128,128)
    model = ResModelNd(3,[128,128],[('conv',32,3,1,1,'relu'),[('conv',32,3,1,1,'relu'),('conv',32,3,1,1,'relu')],[('conv',32,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2)])
    print(model)
    y = model(x)
    print(y)
    print(x.shape,y.shape,model.cnn_output_shape)

def test_res3d():
    x = torch.randn(64,3,32,32,32)
    model = ResModelNd(3,[32,32,32],[('conv',32,3,1,1,'relu'),[('conv',32,3,1,1,'relu'),('conv',32,3,1,1,'relu')],[('conv',32,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],('maxpool',2,2),('conv',64,3,1,1,'relu'),[('conv',64,3,1,1,'relu'),('conv',64,3,1,1,'relu')],[('conv',64,3,1,1,'relu')],('maxpool',2,2)])
    #print(model)
    y = model(x)
    #print(y)
    print(x.shape,y.shape,model.cnn_output_shape)

if __name__=='__main__':
   test_res3d() 
    
