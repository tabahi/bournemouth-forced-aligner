
import torch
import torch.nn as nn

class ModelUtils:
    def __init__(self, layer_dims, input_wav_length, frames_per_window):
        self.layer_dims = layer_dims
        self.input_wav_length = input_wav_length
        self.frames_per_window = frames_per_window

    
    
    @staticmethod
    def calculate_layer_sizes(layer_dims, input_sizes, layer_number=-1):
        # Ensure input_sizes is a tensor
        if not isinstance(input_sizes, torch.Tensor):
            input_sizes = torch.tensor(input_sizes)

        # Ensure input_sizes is 2D: [batch_size, input_size]
        if input_sizes.dim() == 1:
            input_sizes = input_sizes.unsqueeze(0)

        current_sizes = input_sizes

        for i, (kernel_size, stride, padding) in enumerate(layer_dims):
            output_sizes = ((current_sizes + 2 * padding - kernel_size) // stride) + 1
            
            if i == layer_number:
                return output_sizes

            current_sizes = output_sizes

        # If layer_number is -1 or greater than the number of layers, return the last layer's output
        return current_sizes
    
    @staticmethod
    def extract_layer_dims(model):
        layer_dims = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                kernel_size = module.kernel_size[0]
                stride = module.stride[0]
                padding = module.padding[0]
                layer_dims.append((kernel_size, stride, padding))
        return layer_dims

    def get_receptive_field(self):
        """Calculate receptive field size in milliseconds at 16kHz sampling rate"""
        rf_samples = 1
        total_stride = 1
        for kernel, stride, _ in reversed(self.layer_dims):
            rf_samples = rf_samples * stride + (kernel - stride)
            total_stride *= stride
        rf_ms = (rf_samples / 16000) * 1000
        return {
            'rf_ms': rf_ms,
            'rf_samples': rf_samples,
            'total_stride': total_stride,
            'total_downsample_factor': total_stride
        }

    def print_model_info(self):
        """Print comprehensive information about the model's temporal characteristics"""
        input_size_seconds = (self.input_wav_length / 16000)
        frames_per_second = (self.frames_per_window / input_size_seconds).item()
        ms_per_frame = 1000 / frames_per_second
        rf_info = self.get_receptive_field()
        print("\n=== Model Temporal Characteristics ===")
        print("\nInput Window:")
        print(f"• Window size: {input_size_seconds * 1000:.1f}ms ({self.input_wav_length} samples)")
        print("\nTemporal Resolution:")
        print(f"• Frames per window: {self.frames_per_window.item()}")
        print(f"• Frames per second: {frames_per_second:.1f} fps")
        print(f"• Time per frame: {ms_per_frame:.1f}ms")
        print("\nReceptive Field:")
        print(f"• Duration: {rf_info['rf_ms']:.1f}ms")
        print(f"• Samples: {rf_info['rf_samples']} samples")
        print(f"• Total downsampling factor: {rf_info['total_downsample_factor']}")
        print("\nLayer-wise Analysis:")
        curr_rf = 1
        curr_stride = 1
        for i, (kernel, stride, _) in enumerate(reversed(self.layer_dims)):
            curr_rf = curr_rf * stride + (kernel - stride)
            curr_stride *= stride
            ms_rf = (curr_rf / 16000) * 1000
            ms_stride = (curr_stride / 16000) * 1000
            print(f"• Layer {len(self.layer_dims) - i}: RF={ms_rf:.1f}ms, Stride={curr_stride} samples ({ms_stride:.1f}ms)")

    def get_model_info(self):
        """Return comprehensive information about the model's temporal characteristics as a string"""
        input_size_seconds = (self.input_wav_length / 16000)
        frames_per_second = (self.frames_per_window / input_size_seconds).item()
        ms_per_frame = 1000 / frames_per_second
        rf_info = self.get_receptive_field()
        
        info = []
        info.append("\n=== Model Temporal Characteristics ===")
        info.append("Input Window:")
        info.append(f"• Window size: {input_size_seconds * 1000:.1f}ms ({self.input_wav_length} samples)")
        info.append("\nTemporal Resolution:")
        info.append(f"• Frames per window: {self.frames_per_window.item()}")
        info.append(f"• Frames per second: {frames_per_second:.1f} fps")
        info.append(f"• Time per frame: {ms_per_frame:.1f}ms")
        info.append("\nReceptive Field:")
        info.append(f"• Duration: {rf_info['rf_ms']:.1f}ms")
        info.append(f"• Samples: {rf_info['rf_samples']} samples")
        info.append(f"• Total downsampling factor: {rf_info['total_downsample_factor']}")
        info.append("\nLayer-wise Analysis:")
        
        curr_rf = 1
        curr_stride = 1
        for i, (kernel, stride, padding) in enumerate(reversed(self.layer_dims)):
            curr_rf = curr_rf * stride + (kernel - stride)
            curr_stride *= stride
            ms_rf = (curr_rf / 16000) * 1000
            ms_stride = (curr_stride / 16000) * 1000
            invert_i = len(self.layer_dims) - i
            input_dim = self.calculate_layer_sizes(self.layer_dims, self.input_wav_length, layer_number=invert_i-2) if (invert_i>1) else self.input_wav_length
            output_dim = self.calculate_layer_sizes(self.layer_dims, self.input_wav_length, layer_number=invert_i-1)
            
            info.append(f"• Layer {invert_i}: \t  Input={input_dim}, Output={output_dim},\t k={kernel}, s={stride}, p={padding},\t  TR={ms_stride:.1f}ms, RF={ms_rf:.1f}ms ")

        
        return "\n".join(info)



