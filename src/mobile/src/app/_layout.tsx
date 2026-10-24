import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { RepProvider } from '../../lib/rep-store';
import { colors } from '../../lib/theme';
import { useReducedMotion } from '../../components/controls';

export default function Layout() {
  const reduced = useReducedMotion();
  return <RepProvider>
    <StatusBar style="light" />
    <Stack screenOptions={{ headerStyle: { backgroundColor: colors.background }, headerTintColor: colors.green,
      headerTitleStyle: { color: colors.text }, headerShadowVisible: false, headerBackButtonDisplayMode: 'minimal',
      contentStyle: { backgroundColor: colors.background }, animation: reduced ? 'none' : 'slide_from_right' }}>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="record" options={{ title: 'Record a rep' }} />
      <Stack.Screen name="library" options={{ title: 'Saved reps' }} />
      <Stack.Screen name="rep/[name]" options={{ title: 'Replay', presentation: 'modal' }} />
    </Stack>
  </RepProvider>;
}
