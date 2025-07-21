import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Attention, Add, Dropout, Masking, GlobalAveragePooling1D, Concatenate

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)  # (max_len, 3)

    x = Masking(mask_value=0.0)(inputs)
    
    gru_out = GRU(64, return_sequences=True)(x)
    gru_out = GRU(64, return_sequences=True)(gru_out)
    gru_out = Dropout(0.2)(gru_out)

    attention_out = Attention()([gru_out, gru_out])
    context = Add()([gru_out, attention_out])

    # Gộp toàn bộ chuỗi thành 1 vector đại diện
    x = GlobalAveragePooling1D()(context)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Example
input_shape = (100, 3)  
num_classes = 4  
model = build_model(input_shape, num_classes)

model.summary()
