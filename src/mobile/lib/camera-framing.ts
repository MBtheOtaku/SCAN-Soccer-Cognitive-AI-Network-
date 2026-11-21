// Expo's zoom is normalized to the device range, not an optical magnification.
export function clampCameraZoom(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, Math.round(value * 1000) / 1000)) : 0;
}

export function cameraPreviewSize(windowWidth: number, windowHeight: number, expanded: boolean) {
  const availableWidth = Math.max(1, windowWidth - 48);
  // SCAN records portrait 16:9 video. Preserve its 9:16 preview instead of
  // filling the old 4:3 box, which hid the top and bottom of the camera image.
  const height = Math.min(availableWidth * 16 / 9, expanded ? Infinity : Math.max(180, windowHeight - 400));
  return { width: height * 9 / 16, height };
}

export const standardLens = 'AVCaptureDeviceTypeBuiltInWideAngleCamera';
export const ultraWideLens = 'AVCaptureDeviceTypeBuiltInUltraWideCamera';

export function framingLenses(lenses: readonly { deviceType: string }[]) {
  return [
    { id: '', label: 'Default' },
    ...(lenses.some(lens => lens.deviceType === standardLens) ? [{ id: standardLens, label: 'Standard' }] : []),
    ...(lenses.some(lens => lens.deviceType === ultraWideLens) ? [{ id: ultraWideLens, label: 'Ultra wide' }] : []),
  ];
}
