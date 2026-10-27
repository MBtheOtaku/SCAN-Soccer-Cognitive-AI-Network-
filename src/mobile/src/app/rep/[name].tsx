import { useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { useVideoPlayer, VideoView } from 'expo-video';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useEventListener } from 'expo';
import { Action, BallProgress } from '../../../components/controls';
import { RepTitleEditor } from '../../../components/rep-title-editor';
import { RepExport } from '../../../components/rep-export';
import { repDate, useReps } from '../../../lib/rep-store';
import { ui } from '../../../lib/theme';
import type { Rep } from '../../../lib/reps';

function Player({ rep }: { rep: Rep }) {
  const player = useVideoPlayer(rep.uri);
  const [status, setStatus] = useState(player.status);
  const [error, setError] = useState<string | null>(null);
  useEventListener(player, 'statusChange', event => {
    setStatus(event.status); setError(event.error?.message ?? null);
  });
  return <>
    <View style={s.frame}><VideoView player={player} style={s.video} nativeControls contentFit="contain" /></View>
    {(status === 'loading' || status === 'idle') && <BallProgress label="Opening your clip…" />}
    {status === 'error' && <Text style={ui.error}>{error || 'This clip could not be played.'}</Text>}
  </>;
}

export default function Replay() {
  const { name } = useLocalSearchParams<{ name: string }>();
  const { reps, loading } = useReps();
  // Resolve only a saved file name, never an arbitrary URI from a route parameter.
  const rep = reps.find(item => item.name === name);
  return <SafeAreaView style={ui.screen} edges={['bottom', 'left', 'right']}>
    <ScrollView contentContainerStyle={ui.content} keyboardShouldPersistTaps="handled" automaticallyAdjustKeyboardInsets>
      <Text style={ui.eyebrow}>TAKE ANOTHER LOOK</Text>
      <Text style={ui.title}>{rep?.title || 'Every rep\nhas a story.'}</Text>
      {loading ? <BallProgress label="Opening saved reps…" /> : rep ? <>
        <Text style={ui.body}>{repDate(rep.name).toLocaleString()}</Text>
        <RepTitleEditor key={rep.name} rep={rep} />
        <Player rep={rep} />
        <RepExport key={`export-${rep.name}`} rep={rep} />
        <View style={ui.card}><Text style={ui.badge}>SAVED ON THIS PHONE</Text>
          <Text style={ui.body}>Review your movement at your own pace. Coaching analysis is not connected yet.</Text></View>
      </> : <Text style={ui.error}>This rep is not available in your saved library.</Text>}
      <Action title="Back to saved reps" secondary onPress={() => router.canGoBack() ? router.back() : router.replace('/library')} />
    </ScrollView>
  </SafeAreaView>;
}
const s = StyleSheet.create({ frame: { backgroundColor: '#000', borderRadius: 22, overflow: 'hidden' },
  video: { width: '100%', aspectRatio: 3 / 4 } });
