import { useEffect, useState } from 'react';
import { AccessibilityInfo, Animated, Easing, Pressable, StyleSheet, Text, View } from 'react-native';
import { colors, ui } from '../lib/theme';

export function Action({ title, onPress, disabled = false, secondary = false }: {
  title: string; onPress: () => void; disabled?: boolean; secondary?: boolean;
}) {
  return <Pressable accessibilityRole="button" accessibilityState={{ disabled }} disabled={disabled} onPress={onPress}
    style={({ pressed }) => [s.button, secondary && s.secondary, disabled && s.disabled, pressed && s.pressed]}>
    <Text style={[s.buttonText, secondary && { color: colors.text }]}>{title}</Text>
    <Text accessible={false} style={[s.arrow, secondary && { color: colors.green }]}>↗</Text>
  </Pressable>;
}

export function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    let mounted = true;
    void AccessibilityInfo.isReduceMotionEnabled().then(value => { if (mounted) setReduced(value); });
    const listener = AccessibilityInfo.addEventListener('reduceMotionChanged', setReduced);
    return () => { mounted = false; listener.remove(); };
  }, []);
  return reduced;
}

export function Ball({ size = 24 }: { size?: number }) {
  return <View accessible={false} style={{ width: size, height: size, borderRadius: size / 2,
    backgroundColor: '#F2F7EF', borderWidth: 1, borderColor: '#111D14', alignItems: 'center', justifyContent: 'center' }}>
    <View style={{ width: size * .28, height: size * .28, borderRadius: 2, backgroundColor: '#111D14', transform: [{ rotate: '18deg' }] }} />
    {[0, 72, 144, 216, 288].map(angle => <View key={angle} style={{ position: 'absolute', width: size * .17,
      height: size * .17, borderRadius: 2, backgroundColor: '#111D14',
      transform: [{ rotate: `${angle}deg` }, { translateY: -size * .35 }] }} />)}
  </View>;
}

// A number represents known progress. Omit it when there is no measured percentage.
export function BallProgress({ progress, label }: { progress?: number; label: string }) {
  const [value] = useState(() => new Animated.Value(0));
  const [width, setWidth] = useState(0);
  const reduced = useReducedMotion();
  const fraction = progress === undefined ? undefined : Math.max(0, Math.min(1, progress));
  useEffect(() => {
    value.stopAnimation();
    if (fraction !== undefined) {
      const animation = Animated.timing(value, { toValue: fraction, duration: reduced ? 0 : 160, useNativeDriver: false });
      animation.start(); return () => animation.stop();
    }
    value.setValue(reduced ? .5 : 0);
    if (reduced) return;
    const animation = Animated.loop(Animated.sequence([
      Animated.timing(value, { toValue: 1, duration: 1300, easing: Easing.inOut(Easing.quad), useNativeDriver: false }),
      Animated.timing(value, { toValue: 0, duration: 1300, easing: Easing.inOut(Easing.quad), useNativeDriver: false }),
    ]));
    animation.start(); return () => animation.stop();
  }, [fraction, reduced, value]);
  const percent = fraction === undefined ? undefined : Math.round(fraction * 100);
  return <View accessibilityRole="progressbar" accessibilityLabel={label}
    accessibilityValue={percent === undefined ? { text: label } : { min: 0, max: 100, now: percent }}>
    <View style={ui.row}><Text style={ui.small}>{label}</Text>{percent !== undefined && <Text style={s.percent}>{percent}%</Text>}</View>
    <View style={s.trackSpace} onLayout={e => setWidth(Math.max(0, e.nativeEvent.layout.width - 24))}>
      <View style={s.track}>
        <Animated.View style={[s.fill, { width: Animated.multiply(value, width) }]} />
        <Animated.View style={{ position: 'absolute', top: -10, left: -12, transform: [
          { translateX: Animated.multiply(value, width) },
          { rotate: value.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] }) },
        ] }}><Ball /></Animated.View>
      </View>
    </View>
  </View>;
}

const s = StyleSheet.create({
  button: { backgroundColor: colors.green, borderRadius: 16, minHeight: 58, paddingHorizontal: 20,
    paddingVertical: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 12 },
  secondary: { backgroundColor: colors.raised, borderWidth: 1, borderColor: colors.line },
  buttonText: { color: colors.dark, fontSize: 16, fontWeight: '800', flexShrink: 1 },
  arrow: { color: colors.dark, fontSize: 24 }, disabled: { opacity: .42 }, pressed: { opacity: .78 },
  percent: { color: colors.green, fontSize: 13, fontWeight: '800', fontVariant: ['tabular-nums'] },
  trackSpace: { paddingVertical: 22, paddingHorizontal: 12 },
  track: { height: 4, backgroundColor: colors.line, borderRadius: 2 },
  fill: { height: 4, borderRadius: 2, backgroundColor: colors.green },
});
