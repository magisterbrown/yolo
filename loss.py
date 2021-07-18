import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self,coord: float,noobj: float):
        super().__init__()
        self.coord = coord
        self.noobj = noobj
        self.loss = nn.MSELoss()

    def forward(self,input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        blocks = target.shape[-1]
        mask = torch.flatten(target[:,4,:,:]) > 0
        empty = torch.flatten(input[:,4,:,:])[~mask]
        target = self._reflat(target)[mask]
        input = self._reflat(input)[mask]
        for i in range(target.shape[0]):
            target[i,4]=self._iou(input[i,:4],target[i,:4],blocks)
        
        
        target[:,2:4] = torch.sqrt(target[:,2:4])
        input[:,2:4] = torch.sqrt(input[:,2:4])

        loss = self.coord*self._sse(input[:,:2],target[:,:2])
        loss += self.coord*self._sse(input[:,2:4],target[:,2:4])
        loss += self._sse(input[:,4],target[:,4])
        loss += self.noobj*torch.sum(torch.square(empty))
        loss += self._sse(input[:,5:],target[:,5:])
        return loss

    @staticmethod
    def _iou(b1: torch.Tensor,b2: torch.Tensor,blocks: int) -> torch.Tensor:
        b1 = b1.detach()
        b2 = b2.detach()
        cen1 = b1[:2]/blocks
        cen2 = b2[:2]/blocks

        x1a=cen1[0]-b1[2]/2
        x2a=cen1[0]+b1[2]/2
        y1a=cen1[1]-b1[3]/2
        y2a=cen1[1]+b1[3]/2

        x1b=cen2[0]-b2[2]/2
        x2b=cen2[0]+b2[2]/2
        y1b=cen2[1]-b2[3]/2
        y2b=cen2[1]+b2[3]/2

        dx = min(x2a,x2b) - max(x1a,x1b)
        dy = min(y2a,y2b) - max(y1a,y1b)

        if(dx<=0 or dy<=0):
            return 0

        intersection = dx*dy
        union = b1[2]*b1[3]+b2[2]*b2[3]-intersection
        return float(intersection/union)

    @staticmethod
    def _sse(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
        out = torch.square(input-target)
        return torch.sum(out)

    @staticmethod
    def _reflat(pred: torch.Tensor) -> torch.Tensor:
        pred = torch.flatten(pred.transpose(0,1),start_dim=1).transpose(0,1)
        return pred
