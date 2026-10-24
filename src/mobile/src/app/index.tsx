import { router } from 'expo-router';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Action, Ball, BallProgress } from '../../components/controls';
import { repDate, useReps } from '../../lib/rep-store';
import { colors, ui } from '../../lib/theme';

export default function Home() {
  const { reps, loading, pending, error } = useReps();
  const today = new Date().toDateString();
  const todayCount = reps.filter(rep => repDate(rep.name).toDateString() === today).length;
  return <SafeAreaView style={ui.screen} edges={['top', 'left', 'right']}>
    <ScrollView contentContainerStyle={ui.content}>
      <View style={ui.row}>
        <View style={s.brand}><Image source={require('../../assets/scan-icon.png')} style={s.logo} />
          <View><Text style={s.wordmark}>SCAN</Text><Text style={ui.small}>Mind in play.</Text></View></View>
        <View style={s.localBadge}><View style={s.dot} /><Text style={ui.badge}>ON DEVICE</Text></View>
      </View>
      <View style={s.hero}>
        <View accessible={false} style={s.field}>
          <View style={s.fieldLine} /><View style={s.centerCircle} /><View style={s.box} />
          <View style={s.heroBall}><Ball size={50} /></View>
        </View>
        <Text style={ui.eyebrow}>YOUR TRAINING SPACE</Text>
        <Text style={s.heroTitle}>One rep.{ '\n' }A little sharper.</Text>
        <Text style={s.heroCopy}>Find your frame. Capture your movement. Come back and review it.</Text>
        <Action title="Start a rep" disabled={loading} onPress={() => router.push('/record')} />
      </View>
      {loading ? <BallProgress label="Opening your saved reps…" /> : <View style={s.stats}>
        <View style={s.stat}><Text style={s.statNumber}>{todayCount.toString().padStart(2, '0')}</Text><Text style={ui.small}>REPS TODAY</Text></View>
        <View style={s.statDivider} />
        <View style={s.stat}><Text style={s.statNumber}>{reps.length.toString().padStart(2, '0')}</Text><Text style={ui.small}>SAVED ON PHONE</Text></View>
      </View>}
      <Pressable accessibilityRole="button" onPress={() => router.push('/library')}
        style={({ pressed }) => [ui.card, pressed && { opacity: .75 }]}>
        <View style={ui.row}><Text style={ui.eyebrow}>YOUR REPLAY ROOM</Text><Text style={s.chevron}>↗</Text></View>
        <Text style={ui.heading}>Saved reps</Text>
        <Text style={ui.body}>Every saved rep, ready when you are.</Text>
        {reps[0] && <Text style={ui.small}>LATEST · {repDate(reps[0].name).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</Text>}
      </Pressable>
      {(pending || error) && <View style={ui.card}>
        <Text style={ui.error}>{error || 'A recording is waiting to finish saving.'}</Text>
        <Action title="Review saved reps" secondary onPress={() => router.push('/library')} />
      </View>}
      <Text style={ui.small}>Local recording is ready. Coaching analysis will be connected in a later update.</Text>
    </ScrollView>
  </SafeAreaView>;
}

const s = StyleSheet.create({
  brand: { flexDirection: 'row', gap: 10, alignItems: 'center' }, logo: { width: 44, height: 44, borderRadius: 12 },
  wordmark: { color: colors.text, fontSize: 22, fontWeight: '900', letterSpacing: 3 },
  localBadge: { flexDirection: 'row', gap: 6, alignItems: 'center', padding: 10, borderRadius: 20, backgroundColor: colors.raised },
  dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: colors.green },
  hero: { padding: 24, paddingTop: 142, gap: 18, borderRadius: 28, backgroundColor: colors.panel,
    borderWidth: 1, borderColor: colors.line, overflow: 'hidden' },
  heroTitle: { color: colors.text, fontSize: 39, lineHeight: 42, fontWeight: '800', letterSpacing: -1.8 },
  heroCopy: { color: colors.muted, fontSize: 15, lineHeight: 23, maxWidth: 260 },
  field: { position: 'absolute', top: -32, right: -28, width: 260, height: 190,
    borderWidth: 1, borderColor: '#35583C', borderRadius: 14, transform: [{ rotate: '-14deg' }] },
  fieldLine: { position: 'absolute', left: 130, height: '100%', borderLeftWidth: 1, borderColor: '#35583C' },
  centerCircle: { position: 'absolute', top: 57, left: 91, width: 78, height: 78, borderWidth: 1, borderColor: '#35583C', borderRadius: 39 },
  box: { position: 'absolute', left: 0, top: 42, width: 44, height: 104, borderWidth: 1, borderColor: '#35583C' },
  heroBall: { position: 'absolute', top: 87, left: 67, transform: [{ rotate: '14deg' }] },
  stats: { flexDirection: 'row', paddingVertical: 8 }, stat: { flex: 1, gap: 7 },
  statNumber: { color: colors.green, fontSize: 35, fontWeight: '700', fontVariant: ['tabular-nums'] },
  statDivider: { width: 1, backgroundColor: colors.line, marginRight: 24 },
  chevron: { color: colors.green, fontSize: 27 },
});
