import torch
from torch.nn import CTCLoss

from argus.model import Model
from argus.utils import deep_to

from cnd.ocr.model import CRNN
from cnd.ocr.converter import strLabelConverter


class CRNNModel(Model):
    nn_module = {"CRNN": CRNN}
    loss = CTCLoss

    def __init__(self, params):
        super().__init__(params)
        self.converter = strLabelConverter(params["alphabet"])

    def prepare_batch(self, batch, device):
        images, texts = batch["image"], batch["text"]
        images = torch.squeeze(images, 1)
        output = (deep_to(images, device, non_blocking=True), texts)
        return output

    def train_step(self, batch, state) -> dict:
        if not self.nn_module.training:
            self.nn_module.train()

        self.optimizer.zero_grad()

        images, texts = self.prepare_batch(batch, self.device) #TODO: MOVE YOUR BATCH TO DEVICE
        text, length = self.converter.encode(texts)  # USE CONVERTER TO CONVERT TEXT TO INTs and MOVE TO DEVICE TEXT

        preds = self.nn_module(images)

        sim_preds, preds_size = self.preds_converter(preds, len(images))  # CONVERT PREDICTIONS TO TEXT
        loss = self.loss(preds, text, preds_size, length)  # here ctc loss
        self.nn_module.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.nn_module.parameters(), 10)
        self.optimizer.step()

        return {
            "prediction": sim_preds,
            "target": texts,
            "loss": loss.item(),
        }

    def val_step(self, batch, state) -> dict:
        if self.nn_module.training:
            self.nn_module.eval()

        with torch.no_grad():
            #TODO: COMPUTE JUST THE SAME WAY AS TRAIN, BUT WITH NO GRAD!
            self.optimizer.zero_grad()

            images, texts = self.prepare_batch(batch, self.device)  # TODO: MOVE YOUR BATCH TO DEVICE
            text, length = self.converter.encode(texts)  # USE CONVERTER TO CONVERT TEXT TO INTs and MOVE TO DEVICE TEXT

            preds = self.nn_module(images)

            sim_preds, preds_size = self.preds_converter(preds, len(images))  # CONVERT PREDICTIONS TO TEXT
            loss = self.loss(preds, text, preds_size, length)  # here ctc loss

            return {
                "prediction": sim_preds,
                "target": texts,
                "loss": loss.item()
            }

    def preds_converter(self, logits, len_images):
        preds_size = torch.IntTensor([logits.size(0)] * len_images)
        _, preds = logits.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = self.converter.decode(preds, preds_size, raw=False)

        return sim_preds, preds_size