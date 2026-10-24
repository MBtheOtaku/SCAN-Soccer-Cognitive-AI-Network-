import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Action, BallProgress } from '../../components/controls';
import { repDate, useReps } from '../../lib/rep-store';
import { colors, ui } from '../../lib/theme';

export default function Library() {
  const { reps, loading, pending, saving, error, retry } = useReps();
  return <SafeAreaView style={ui.screen} edges={['bottom', 'left', 'right']}>
    <ScrollView contentContainerStyle={ui.content}>
      <View style={{ gap: 10 }}><Text style={ui.eyebrow}>THE REPLAY ROOM</Text><Text style={ui.title}>Your work,{ '\n' }worth another look.</Text>
        <Text style={ui.body}>{reps.length} {reps.length === 1 ? 'rep' : 'reps'} saved on this phone.</Text></View>
      {loading && <BallProgress label="Opening saved reps…" />}
      {(error || pending) && <View style={ui.card}>
        <Text style={ui.error}>{error || 'One rep is waiting to finish saving.'}</Text>
        {saving ? <BallProgress label="Saving your rep…" /> :
          <Action title="Retry save" secondary onPress={() => { void retry().catch(() => {}); }} />}
      </View>}
      {!loading && !reps.length && <View style={[ui.card, { paddingVertical: 32 }]}>
        <Text style={ui.heading}>Your first rep starts here.</Text>
        <Text style={ui.body}>Record a short clip. Once it is saved, you can return and replay it here.</Text>
        <Action title="Record a rep" onPress={() => router.push('/record')} />
      </View>}
      {reps.map((rep, i) => {
        const date = repDate(rep.name);
        return <Pressable key={rep.name} accessibilityRole="button"
          accessibilityLabel={`Play rep ${reps.length - i}, ${date.toLocaleString()}`}
          onPress={() => router.push({ pathname: '/rep/[name]', params: { name: rep.name } })}
          style={({ pressed }) => [ui.card, s.rep, pressed && { opacity: .7 }]}>
          <View style={s.play}><Text style={s.playIcon}>▶</Text></View>
          <View style={s.description}><Text style={ui.heading}>Rep {String(reps.length - i).padStart(2, '0')}</Text>
            <Text style={ui.small}>{date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} · {date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })}</Text>
            <Text style={ui.badge}>SAVED LOCALLY</Text>
          </View><Text style={s.chevron}>↗</Text>
        </Pressable>;
      })}
      {!!reps.length && <Text style={ui.small}>Saved clips stay in this app. Uninstalling SCAN removes its local recordings.</Text>}
    </ScrollView>
  </SafeAreaView>;
}
const s = StyleSheet.create({
  rep: { flexDirection: 'row', alignItems: 'center', gap: 16 }, description: { flex: 1, gap: 8 },
  play: { width: 48, height: 56, borderRadius: 14, backgroundColor: colors.raised, alignItems: 'center', justifyContent: 'center' },
  playIcon: { color: colors.green, fontSize: 18 }, chevron: { color: colors.green, fontSize: 22 },
});
