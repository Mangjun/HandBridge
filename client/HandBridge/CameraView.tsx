import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { Camera, useCameraDevices } from 'react-native-vision-camera';
import cameraStyles from './CameraView.styles';

export default function CameraView() {
  const [devicePosition, setDevicePosition] = useState<'front' | 'back'>('front');
  const devices = useCameraDevices();
  const cameraRef = useRef<Camera>(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [translatedText, setTranslatedText] = useState('번역된 수어 텍스트 예시');

  useEffect(() => {
    (async () => {
      const camera = await Camera.requestCameraPermission();
      const mic = await Camera.requestMicrophonePermission();
      setHasPermission(camera === 'granted' && mic === 'granted');
    })();
  }, []);

  const device = devices.find((d) => d.position === devicePosition);

  const toggleCamera = () => {
    setDevicePosition(prev => (prev === 'front' ? 'back' : 'front'));
  };

  // 영상 업로드 함수
  const uploadVideoToServer = async (filePath: string) => {
    const file = {
      uri: Platform.OS === 'ios' ? filePath : 'file://' + filePath,
      type: 'video/mp4',
      name: 'sign_video.mp4',
    };

    const formData = new FormData();
    formData.append('file', file as any);

    try {
      const response = await fetch('http://192.168.45.187:8000/api/v1/video/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });
      const result = await response.json();
      console.log('✅ 업로드 성공:', result);
    } catch (error) {
      console.error('❌ 업로드 실패:', error);
    }
  };


  const startRecording = async () => {
    if (!cameraRef.current) return;
    setIsRecording(true);
    await cameraRef.current.startRecording({
      flash: 'off',
      onRecordingFinished: (video) => {
        console.log('녹화 완료:', video);
        setIsRecording(false);
        uploadVideoToServer(video.path);
      },
      onRecordingError: (err) => {
        console.error('녹화 실패:', err);
        setIsRecording(false);
      },
    });
  };

  const stopRecording = async () => {
    if (!cameraRef.current) return;
    await cameraRef.current.stopRecording();
    setIsRecording(false);
  };

  if (!device || !hasPermission) {
    return (
      <View style={cameraStyles.container}>
        <Text style={cameraStyles.warningText}>카메라 권한이 없거나 로딩 중입니다.</Text>
      </View>
    );
  }

  return (
    <View style={cameraStyles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        video={true}
        audio={true}
      />
      <View style={cameraStyles.translationWrapper}>
        <Text style={cameraStyles.translatedText}>{translatedText}</Text>
      </View>
      <View style={cameraStyles.controls}>
        {isRecording ? (
          <TouchableOpacity style={[cameraStyles.button, cameraStyles.stop]} onPress={stopRecording}>
            <Text style={cameraStyles.buttonText}>⏹️ 중지</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={[cameraStyles.button, cameraStyles.record]} onPress={startRecording}>
            <Text style={cameraStyles.buttonText}>⏺️ 녹화</Text>
          </TouchableOpacity>

        )}
        <TouchableOpacity style={[cameraStyles.button]} onPress={toggleCamera}>
          <Text style={cameraStyles.buttonText}>🔁 카메라 전환</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}