import torch
from .tobias_comparator import TobiasComparator
from .tobias_embedder import EmbedderInterface


class TobiasControlModule:
    def __init__(
        self,
        embedder: EmbedderInterface,
        comparator: TobiasComparator,
        tokenizer=None,
        pad_fill=None,
        prompt_fill=None,
        sep_token_id=None,
        pad_token_id=None,
    ):
        self.comparator = comparator
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.pad_fill = pad_fill
        self.prompt_fill = prompt_fill
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

    def calc_weights(self, input_ids, encoder_hidden_states=None, topic_prompt=None):
        if input_ids is None:
            print("input_ids is none")
            return
        word_weights = torch.zeros(input_ids.shape)
        for i in range(input_ids.shape[0]):
            # Replace -100 with pad_token
            tokens = torch.where(
                input_ids[i] != -100, input_ids[i], self.tokenizer.pad_token_id
            )

            # Get token embeddings
            embeddings = self.embedder.embed(
                input_ids=tokens,
                encoder_hidden_states=encoder_hidden_states.last_hidden_state[i],
            )
            topic_prompt_embeds = None
            if topic_prompt is not None:
                topic_prompt_embeds = self.embedder.embed_prompt(topic_prompt[i])

            # Calculate similarity between prompt and document
            word_weights[i] = self.prompt_similarity(
                embeddings=embeddings,
                tokens=tokens,
                topic_prompt_embeddings=topic_prompt_embeds,
            )
        return word_weights.to(input_ids.device)

    def prompt_similarity(
        self,
        embeddings,
        tokens,
        topic_prompt_embeddings=None,
    ):
        token_weights = torch.full(
            tokens.size(),
            fill_value=self.pad_fill,
            device=tokens.device,
            dtype=torch.float32,
        )
        # torch.save(embeddings, 'new_embeddings.tensor')
        # torch.save(tokens, 'new_tok.tensor')

        if topic_prompt_embeddings == None:
            sep_token_index = -1
            for j in range(len(tokens)):
                if tokens[j] == self.sep_token_id:
                    sep_token_index = j
                    break
            # Skip start_token (idx 1)
            prompt_words = embeddings[1:sep_token_index]
            # torch.save(prompt_words, 'new_prompt_words.tensor')
            # Between sep_token (idx sep_token_index+1) and end_token (idx -1)
            # article_embeds = embeddings[sep_token_index + 1 : -1]
            article_embeds = embeddings
            # torch.save(article_embeds, 'new_article_embeds.tensor')

            similarity = self.comparator.compare(prompt_words, article_embeds)
            # torch.save(similarity, 'new_similarity.tensor')

            # Fill token_weights for article_tokens with similarity
            # token_weights[sep_token_index + 1: -1].copy_(similarity)
            token_weights.copy_(similarity)
            if self.prompt_fill is not None:
                token_weights[1:sep_token_index] = torch.full(
                    (sep_token_index - 1,), self.prompt_fill
                )
            token_weights[0] = self.pad_fill
            token_weights[sep_token_index] = self.pad_fill
        else:
            similarity = self.comparator.compare(topic_prompt_embeddings, embeddings)
            token_weights.copy_(similarity)

        # Fill BOS token with pad_fill
        # Fill EOS token with pad_fill
        # Fill PAD token with pad_fill
        token_weights = torch.where(
            torch.isin(tokens, torch.tensor([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id], device=tokens.device)),
            self.pad_fill,
            token_weights,
        )
        return token_weights
