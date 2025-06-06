import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import CameraView from './CameraView'; // 위치에 따라 경로 조정

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <CameraView />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});