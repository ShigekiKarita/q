import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env, resize=84, strip=False, gray=True):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    if strip:
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    if resize > 0:
        fun = T.Compose([T.ToPILImage(),
                         T.Resize((resize, resize), interpolation=Image.CUBIC),
                         T.ToTensor()])
        screen = fun(screen)
    if gray:
        screen = 0.216 * screen[0] + 0.7152 * screen[1] + 0.0722 * screen[2]
        screen = screen.unsqueeze(0)
    return screen.unsqueeze(0)


class FrameWindow:
    def __init__(self, n_frames, init_frame):
        """
        n_frames (int): number of frames
        init_frame (tensor): initial frame with shape (B=1, C, H, W)
        """
        self.assert_shape(init_frame)
        self.n_frames = n_frames
        self.data = torch.zeros(n_frames, *init_frame.shape, dtype=init_frame.dtype, device=init_frame.device)
        self.data[:] = init_frame

    def assert_shape(self, frame):
        assert frame.dim() == 4
        assert frame.shape[0] == 1

    def insert_(self, new_frame):
        self.data[1:] = self.data[0:-1]
        self.data[0] = new_frame

    def as_state(self):
        """
        return (torch.tensor): (1, n_frames * C, H, W)
        """
        return self.data.view(1, self.n_frames * self.data.shape[2], *self.data.shape[3:])

