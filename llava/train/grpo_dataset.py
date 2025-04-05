class LLaVAGRPODataset(Dataset):
    def __init__(self, dataset, image_folder, tokenizer, image_processor):
        self.dataset = dataset
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        print(f"Loaded {len(dataset)} examples for GRPO training")
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load and process image
        if isinstance(item['image'], str):
            image_file = item['image']
            if not os.path.isabs(image_file):
                image_file = os.path.join(self.image_folder, image_file)
            image = Image.open(image_file).convert('RGB')
        else:
            image = item['image']
        
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        return {
            'input_ids': item['input_ids'] if 'input_ids' in item else None,
            'query': item['query'] if 'query' in item else None,
            'answer': item['answer'] if 'answer' in item else None,
            'image': image_tensor,
        }
