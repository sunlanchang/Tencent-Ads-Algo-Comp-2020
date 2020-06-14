def get_age_model(creative_id_emb, ad_id_emb, product_id_emb):
    embed_dim = 128  # Embedding size for each token
    num_heads = 1  # Number of attention heads
    ff_dim = 256  # Hidden layer size in feed forward network inside transformer

    # shape：(sequence长度, )
    # first input
    input_creative_id = Input(shape=(None,), name='creative_id')
    x1 = TokenAndPositionEmbedding(
        maxlen, NUM_creative_id, embed_dim, creative_id_emb)(input_creative_id)
    for _ in range(args.num_transformer):
        x1 = TransformerBlock(embed_dim, num_heads, ff_dim)(x1)
    for _ in range(args.num_lstm):
        x1 = Bidirectional(LSTM(256, return_sequences=True))(x1)
    x1 = layers.GlobalMaxPooling1D()(x1)

    # second input
    input_ad_id = Input(shape=(None,), name='ad_id')
    x2 = TokenAndPositionEmbedding(
        maxlen, NUM_ad_id, embed_dim, ad_id_emb)(input_ad_id)
    for _ in range(args.num_transformer):
        x2 = TransformerBlock(embed_dim, num_heads, ff_dim)(x2)
    for _ in range(args.num_lstm):
        x2 = Bidirectional(LSTM(256, return_sequences=True))(x2)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # third input
    input_product_id = Input(shape=(None,), name='product_id')
    x3 = TokenAndPositionEmbedding(
        maxlen, NUM_product_id, embed_dim, product_id_emb)(input_product_id)
    for _ in range(args.num_transformer):
        x3 = TransformerBlock(embed_dim, num_heads, ff_dim)(x3)
    for _ in range(args.num_lstm):
        x3 = Bidirectional(LSTM(256, return_sequences=True))(x3)
    x3 = layers.GlobalMaxPooling1D()(x3)

    # concat x1 x2 x3
    x = concatenate([x1, x2, x3])
    # x = x1 + x2 + x3
    x = Dense(20)(x)
    output_y = Dense(10, activation='softmax')(x)

    model = Model([input_creative_id, input_ad_id, input_product_id], output_y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model
