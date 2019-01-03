from torch import nn


class DQN(nn.Module):

    def __init__(self, c, h, w, n_action, n_hidden=64, n_conv=2, kernel_size=3, stride=2):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, n_hidden, kernel_size, stride),
            nn.BatchNorm2d(n_hidden),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Conv2d(n_hidden, n_hidden, kernel_size, stride),
                    nn.BatchNorm2d(n_hidden),
                    nn.ReLU(inplace=True)
                )
                for _ in range(n_conv)
            ]
        )

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size):
            for _ in range(n_conv + 1):
                size = (size - (kernel_size - 1) - 1) // stride  + 1
            return size
        convw = conv2d_size_out(w)
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * n_hidden
        self.head = nn.Linear(linear_input_size, n_action) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.conv(x)
        return self.head(x.view(x.size(0), -1))
