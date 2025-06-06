import { StyleSheet } from 'react-native';

const cameraStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  warningText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 40,
  },
  controls: {
    position: 'absolute',
    bottom: 30,
    width: '100%',
    paddingHorizontal: 24,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    backgroundColor: '#333',
    borderRadius: 8,
    minWidth: 130,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
  record: {
    backgroundColor: 'crimson',
  },
  stop: {
    backgroundColor: 'gray',
  },
  translationWrapper: {
    position: 'absolute',
    bottom: 90,
    width: '100%',
    alignItems: 'center',
    paddingHorizontal: 16,
    zIndex: 10,
  },
  translatedText: {
    color: 'white',
    fontSize: 18,
    textAlign: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    width: '100%',
  },
});

export default cameraStyles;