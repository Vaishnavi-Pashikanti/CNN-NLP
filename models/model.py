from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", cache_dir="./models")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", cache_dir="./models")
