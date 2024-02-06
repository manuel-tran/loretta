class LoReTTa:
    """
    Pseudo-code for commutative and transitive modeling
    """
    
    def forward(self, tokens, modes=['commutative','transitive']):
        """
        tokens ... tokenized inputs, e.g., [x_0,...x_n, y_0,...,y_m]
        x_0, y_0 .... modality-specific tokens, 'a', 'b', or 'c'
        """
        
        if 'commutative' in modes: #shuffle modalities
            tokens = self.shuffle_modalities(tokens)
          
        if 'transitive' in modes: #generate missing modality
            existing_modalities = self.extract_modality_tokens(tokens)
          
            if ['a', 'b'] in existing_modalities: #case 1
                modality_a, modality_b = self.split_tokens(tokens)
                modality_c = self.model.generate([modality_b, 'c'])
                tokens = [modality_c, modality_a]
              
            if ['b', 'c'] in existing_modalities: #case 2
                modality_b, modality_c = self.split_tokens(tokens)
                modality_a = self.model.generate([modality_b, 'a'])
                tokens = [modality_a, modality_c]
              
            if ['a'] in existing_modalities \ #case 3
                and len(existing_modalities) == 1: #edge case with on modality
                modality_b = self.model.generate([modality_a, 'b'])
                modality_c = self.model.generate([modality_b, 'c'])
                tokens = [modality_c, modality_a]
          
            if ['b'] in existing_modalities \ #case 4
                and len(existing_modalities) == 1: #edge case with on modality
                modality_a = self.model.generate([modality_b, 'a'])
                modality_c = self.model.generate([modality_b, 'c'])
                tokens = self.shuffle_modalities([modality_a, modality_c])
          
            if ['c'] in existing_modalities \ #case 5
                and len(existing_modalities) == 1: #edge case with on modality
                modality_b = self.model.generate([modality_c, 'b'])
                modality_a = self.model.generate([modality_b, 'a'])
                tokens = [modality_a, modality_c]
          
            if self.prob_use_all_modalities < rand(1): #occcasionally use all modalities
                tokens = self.shuffle_modalities([modality_a, modality_b, modality_c])
              
        logits = self.model(tokens[:, :-1]) #get predictions
        targets = tokens[:, +1:] #shift targets
      
        loss = self.criterion(logits, targets) #calculate cce-loss
        return self.split_loss(loss) #return individual loss for each modality
