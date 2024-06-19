class EmbedderInterface:
    use_encoder_hidden_states = False

    def embed(self, input_ids, encoder_hidden_states=None):
        raise NotImplementedError

    def embed_prompt(self, input_ids):
        raise NotImplementedError


class ContextualEmbedder(EmbedderInterface):
    use_encoder_hidden_states = True

    def __init__(self, model):
        self.encoder = model.model.encoder

    def embed(self, input_ids, encoder_hidden_states):
        return encoder_hidden_states

    def _force_embed(self, input_ids):
        encoder_output = self.encoder(
            input_ids.unsqueeze(0), return_dict=True, skip_hook=True
        )
        return encoder_output.last_hidden_state[0]

    def embed_prompt(self, input_ids):
        return self._force_embed(input_ids)


class InternalEmbedder(EmbedderInterface):
    def __init__(self, model):
        self.embedder = model.model.encoder.embed_tokens

    def embed(self, input_ids, encoder_hidden_states=None):
        return self.embedder(input_ids)

    def embed_prompt(self, input_ids):
        return self.embed(input_ids)


def embedder_factory(embedder_type: str, model=None, tokenizer=None):
    embedder_types = ["internal", "contextual"]

    if embedder_type not in embedder_types:
        raise NameError

    if embedder_type == "internal":
        return InternalEmbedder(model)
    elif embedder_type == "contextual":
        return ContextualEmbedder(model)
