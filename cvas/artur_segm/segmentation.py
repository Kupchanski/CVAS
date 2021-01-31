import torch.nn as nn
import torch.nn.functional as functional

class SegmentationModule(nn.Module):
    _IGNORE_INDEX = 255

    class _MeanFusion:
        def __init__(self, x, classes):
            self.buffer = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.counter = 0

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            self.counter += 1
            self.buffer.add_((probs - self.buffer) / self.counter)

        def output(self):
            probs, cls = self.buffer.max(1)
            return probs, cls

    class _VotingFusion:
        def __init__(self, x, classes):
            self.votes = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))
            self.probs = x.new_zeros(x.size(0), classes, x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            probs, cls = probs.max(1, keepdim=True)

            self.votes.scatter_add_(1, cls, self.votes.new_ones(cls.size()))
            self.probs.scatter_add_(1, cls, probs)

        def output(self):
            cls, idx = self.votes.max(1, keepdim=True)
            probs = self.probs / self.votes.clamp(min=1)
            probs = probs.gather(1, idx)
            return probs.squeeze(1), cls.squeeze(1)

    class _MaxFusion:
        def __init__(self, x, _):
            self.buffer_cls = x.new_zeros(x.size(0), x.size(2), x.size(3), dtype=torch.long)
            self.buffer_prob = x.new_zeros(x.size(0), x.size(2), x.size(3))

        def update(self, sem_logits):
            probs = functional.softmax(sem_logits, dim=1)
            max_prob, max_cls = probs.max(1)

            replace_idx = max_prob > self.buffer_prob
            self.buffer_cls[replace_idx] = max_cls[replace_idx]
            self.buffer_prob[replace_idx] = max_prob[replace_idx]

        def output(self):
            return self.buffer_prob, self.buffer_cls

    def __init__(self, body, head, head_channels, classes, fusion_mode="mean"):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.cls = nn.Conv2d(head_channels, classes, 1)

        self.classes = classes
        if fusion_mode == "mean":
            self.fusion_cls = SegmentationModule._MeanFusion
        elif fusion_mode == "voting":
            self.fusion_cls = SegmentationModule._VotingFusion
        elif fusion_mode == "max":
            self.fusion_cls = SegmentationModule._MaxFusion

    def _network(self, x, scale):
        if scale != 1:
            scaled_size = [round(s * scale) for s in x.shape[-2:]]
            x_up = functional.upsample(x, size=scaled_size, mode="bilinear")
        else:
            x_up = x

        x_up = self.body(x_up)
        x_up = self.head(x_up)
        sem_logits = self.cls(x_up)

        del x_up
        return sem_logits

    def forward(self, x, scales, do_flip=True):
        out_size = x.shape[-2:]
        fusion = self.fusion_cls(x, self.classes)

        for scale in scales:
            # Main orientation
            sem_logits = self._network(x, scale)
            sem_logits = functional.upsample(sem_logits, size=out_size, mode="bilinear")
            fusion.update(sem_logits)

            # Flipped orientation
            if do_flip:
                # Main orientation
                sem_logits = self._network(flip(x, -1), scale)
                sem_logits = functional.upsample(sem_logits, size=out_size, mode="bilinear")
                fusion.update(flip(sem_logits, -1))

        return fusion.output()