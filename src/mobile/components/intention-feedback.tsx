import { useState } from 'react';
import { Linking, Pressable, Text, View } from 'react-native';
import type { IntentionFeedback as Feedback } from '../lib/intention-feedback';
import { shotLabels } from '../lib/training-intention';
import { colors, ui } from '../lib/theme';

export function IntentionFeedback({ feedback }: { feedback: Feedback }) {
  const [details, setDetails] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);
  return <View style={{ gap: 14 }}>
    <Text style={ui.badge}>{shotLabels[feedback.shot_type].toUpperCase()} INTENTION</Text>
    <Text style={ui.body}>{feedback.focus}</Text>
    <Text style={ui.body}>{feedback.summary}</Text>
    {feedback.reported_outcome && <Text style={ui.small}>Player-reported outcome: {feedback.reported_outcome.value} · Not verified by SCAN</Text>}
    {feedback.metrics.map(metric => <View key={metric.key} style={{ gap: 4 }}>
      <Text style={ui.heading}>{metric.label}</Text>
      {metric.status === 'available' && metric.value !== null
        ? <Text style={ui.body}>{metric.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}{metric.unit ? ` ${metric.unit}` : ''}</Text>
        : <Text style={ui.body}>Not measured · {metric.reason}</Text>}
      {details && metric.formula && <Text style={ui.small}>Definition: {metric.formula}</Text>}
    </View>)}
    <Text style={ui.small}>{feedback.reference.note}</Text>
    <Pressable accessibilityRole="button" accessibilityState={{ expanded: details }}
      onPress={() => setDetails(current => !current)} style={[ui.row, { minHeight: 44 }]}>
      <Text style={[ui.small, { color: colors.green }]}>{details ? 'Hide measurement details' : 'Measurement details & research'}</Text>
      <Text style={ui.heading}>{details ? '−' : '+'}</Text>
    </Pressable>
    {details && feedback.reference.profiles.map((profile, index) => <View key={index} style={{ gap: 6 }}>
      <Text style={ui.badge}>RESEARCH COHORT · NO PERSONAL RATING</Text>
      <Text style={ui.body}>{profile.mean} ± {profile.sd} {profile.unit} · {profile.sample_size} participants</Text>
      <Text style={ui.small}>{profile.statistic}</Text>
      <Text style={ui.small}>{profile.cohort}</Text>
      <Text style={ui.small}>{profile.protocol}</Text>
      <Text style={ui.small}>{profile.limitation}</Text>
    </View>)}
    {details && feedback.reference.sources.map(source => <Pressable key={source.url} accessibilityRole="link"
      accessibilityLabel={source.title} style={{ minHeight: 44, justifyContent: 'center' }}
      onPress={() => { setLinkError(null); void Linking.openURL(source.url).catch(() => setLinkError('The research link could not be opened.')); }}>
      <Text style={[ui.small, { color: colors.green, textDecorationLine: 'underline' }]}>{source.title}</Text>
    </Pressable>)}
    {linkError && <Text style={ui.error} accessibilityRole="alert">{linkError}</Text>}
  </View>;
}
