from transformers import SeamlessM4Tv2Model, AutoProcessor

class SeamlessM4T:
    def __init__(self, model_name, device):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(model_name).to(device)
        self.device = device
    
    def forward(self, text):
        input_ids = self.processor(text=[text], return_tensors="pt")
        input_ids = {k: v.to(self.device) for k, v in input_ids.items()}
    
        output_ids = self.model.generate(**input_ids, tgt_lang="urd", generate_speech=False)

        output = self.processor.batch_decode(output_ids[0], skip_special_tokens=True)[0]
        return output
