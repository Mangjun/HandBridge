import torch
import torch.nn as nn

# 수어 단어 분류용 LSTM 모델 정의
class SignModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=100, bidirectional=True, dropout=0.5):
        """
        input_size: 프레임당 feature 수 (keypoints = 126)
        hidden_size: LSTM hidden state 크기
        num_layers: LSTM layer 수
        num_classes: 예측할 단어 라벨 개수
        bidirectional: 양방향 LSTM 여부
        dropout: LSTM dropout 비율
        """
        super(SignModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM 정의
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 양방향이면 hidden size 두 배
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # 최종 분류 layer
        self.classifier = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x, lengths):
        """
        x: [B, T, 126] padded 시퀀스 입력
        lengths: [B] 실제 시퀀스 길이
        """
        # 시퀀스 정렬을 위한 pack 처리
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # LSTM 처리
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # 마지막 hidden state 가져오기
        if self.bidirectional:
            # [num_layers * 2, B, H] → forward + backward hidden state 결합
            h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
            last_hidden = torch.cat((h_n[-1][0], h_n[-1][1]), dim=1)  # [B, 2H]
        else:
            last_hidden = h_n[-1]  # [B, H]

        # 분류기 통과 → [B, num_classes]
        logits = self.classifier(last_hidden)
        return logits
