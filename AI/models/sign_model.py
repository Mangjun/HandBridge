import torch
import torch.nn as nn

# 수어 단어 분류용 LSTM 모델 정의 (Dropout & enforce_sorted=False 추가)
class SignModel(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 100,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        """
        input_size:    프레임당 feature 수 (keypoints = 126)
        hidden_size:   LSTM hidden state 크기
        num_layers:    LSTM layer 수
        num_classes:   예측할 단어 라벨 개수
        bidirectional: 양방향 LSTM 여부
        dropout:       dropout 비율 (LSTM 레이어 + classifier 직후)
        """
        super(SignModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM 정의 (layer 사이 dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # 양방향이면 hidden size 두 배
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Dropout 레이어 추가
        self.dropout = nn.Dropout(p=dropout)

        # 최종 분류 layer
        self.classifier = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, T, input_size] padded 시퀀스 입력
        lengths: [B] 실제 시퀀스 길이
        """
        # pack_padded_sequence: enforce_sorted=False로 정렬 필요 제거
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # LSTM 처리
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # 마지막 hidden state 취합
        if self.bidirectional:
            # [num_layers * 2, B, H] → (layer, dir, batch, hid)
            h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
            # 마지막 레이어의 forward + backward hidden state 결합
            last_hidden = torch.cat((h_n[-1][0], h_n[-1][1]), dim=1)
        else:
            # [B, H]
            last_hidden = h_n[-1]

        # Dropout 적용 후 분류
        dropped = self.dropout(last_hidden)
        logits = self.classifier(dropped)
        return logits
