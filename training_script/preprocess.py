from transformers import BartTokenizer


def preprocess_dataset(
    examples,
    text_column,
    summary_column,
    tokenizer: BartTokenizer,
    padding,
    data_args,
    max_target_length,
    prefix,
    run_args,
):
    inputs, targets = [], []
    topic_prompts = []

    topic_prompt_column = run_args.topic_prompt_column
    sep_token = tokenizer.sep_token

    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            if data_args.preprocessed_input is not None:
                inputs.append(examples[data_args.preprocessed_input][i])
            else:
                topic_prompt = examples[topic_prompt_column][i]
                if run_args.remove_commas_prompt:
                    topic_prompt = " " + " ".join(topic_prompt.split(", ")).strip()

                if run_args.topic_as_prompt:
                    inputs.append(topic_prompt + sep_token + examples[text_column][i])
                else:
                    inputs.append(examples[text_column][i])
                    topic_prompts.append(topic_prompt)

            targets.append(examples[summary_column][i])


    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    if not run_args.topic_as_prompt:
        topic_prompts = tokenizer(
            topic_prompts,
            max_length=data_args.max_source_length,
            padding='longest',
            truncation=True,
        )
        model_inputs["topic_prompt"] = topic_prompts["input_ids"]

    if data_args.preprocessed_weights is not None:
        model_inputs["weights"] = examples[data_args.preprocessed_weights]

    model_inputs["tid"] = examples["tid"]
    return model_inputs


def preprocess_dataset_old(
    examples,
    text_column,
    summary_column,
    tokenizer,
    padding,
    data_args,
    max_target_length,
    prefix,
):
    # remove pairs where at least one record is None

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
