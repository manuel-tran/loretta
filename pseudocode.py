class LoReTTa:
    """
    Pseudo-code for commutative and transitive modeling
    """
    
    def forward(self, tokens, modes=['commutative','transitive']):
        """
        tokens ... tokenized inputs, e.g., [a_0,...,a_n, b_0,...,b_m]
        a_0, b_0 .... modality-specific tokens, e.g., 'a', 'b', or 'c'
        """
        
        if 'commutative' in modes: #shuffle modalities
            tokens = self.shuffle_modalities(tokens)
          
        if 'transitive' in modes: #generate missing modality
            existing_modalities = self.extract_modality_tokens(tokens)
          
            if ['a', 'b'] in existing_modalities: #case 1
                tokens_a, tokens_b = self.split_tokens(tokens)
                tokens_c = self.model.generate([tokens_b, 'c'])
                tokens = [tokens_c, tokens_a]
              
            if ['b', 'c'] in existing_modalities: #case 2
                tokens_b, tokens_c = self.split_tokens(tokens)
                tokens_a = self.model.generate([tokens_b, 'a'])
                tokens = [tokens_a, tokens_c]
              
            if ['a'] in existing_modalities \ #case 3
                and len(existing_modalities) == 1: #edge case with one modality
                tokens_b = self.model.generate([tokens_a, 'b'])
                tokens_c = self.model.generate([tokens_b, 'c'])
                tokens = [tokens_c, tokens_a]
          
            if ['b'] in existing_modalities \ #case 4
                and len(existing_modalities) == 1: #edge case with one modality
                tokens_a = self.model.generate([tokens_b, 'a'])
                tokens_c = self.model.generate([tokens_b, 'c'])
                tokens = self.shuffle_modalities([tokens_a, tokens_c])
          
            if ['c'] in existing_modalities \ #case 5
                and len(existing_modalities) == 1: #edge case with one modality
                tokens_b = self.model.generate([tokens_c, 'b'])
                tokens_a = self.model.generate([tokens_b, 'a'])
                tokens = [tokens_a, tokens_c]
          
            if self.prob_use_all_modalities < rand(1): #occcasionally use all modalities
                tokens = self.shuffle_modalities([tokens_a, tokens_b, tokens_c])
              
        logits = self.model(tokens[:, :-1]) #get predictions
        targets = tokens[:, +1:] #shift targets
      
        loss = self.criterion(logits, targets) #calculate cce-loss
        return self.split_loss(loss) #return individual loss for each modality
