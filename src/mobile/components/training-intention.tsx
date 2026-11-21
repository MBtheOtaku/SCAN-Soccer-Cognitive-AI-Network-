import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { normalizeIntention, outcomeLabels, reportedOutcomes, shotLabels, shotTypes, targetLabels, targetZones,
  type TrainingIntention } from '../lib/training-intention';
import { colors, ui } from '../lib/theme';

function Choice({ label, selected, disabled, onPress }: {
  label: string; selected: boolean; disabled: boolean; onPress: () => void;
}) {
  return <Pressable accessibilityRole="button" accessibilityLabel={label}
    accessibilityState={{ selected, disabled }} disabled={disabled} onPress={onPress}
    style={[s.choice, selected && s.selected, disabled && { opacity: .45 }]}>
    <Text style={[s.label, selected && { color: colors.dark }]}>{label}</Text>
  </Pressable>;
}

export function TrainingIntentionFields({ value, onChange, disabled = false, includeOutcome = false }: {
  value: TrainingIntention; onChange: (value: TrainingIntention) => void; disabled?: boolean; includeOutcome?: boolean;
}) {
  const intention = normalizeIntention(value);
  const [expanded, setExpanded] = useState(Boolean(intention.target_zone || intention.reported_outcome));
  return <View style={{ gap: 12 }}>
    <Text style={ui.badge}>TRAINING INTENTION</Text>
    <Text style={ui.small}>What are you practising in this rep?</Text>
    <View style={s.choices}>{shotTypes.map(shot => <Choice key={shot} label={shotLabels[shot]}
      selected={intention.shot_type === shot} disabled={disabled}
      onPress={() => onChange({ ...intention, shot_type: shot })} />)}</View>
    <Pressable accessibilityRole="button" accessibilityState={{ expanded, disabled }} disabled={disabled}
      onPress={() => setExpanded(current => !current)} style={[ui.row, { minHeight: 44 }]}>
      <Text style={ui.small}>{includeOutcome ? 'Target & your outcome (optional)' : 'Target area (optional)'}</Text>
      <Text style={ui.heading}>{expanded ? '−' : '+'}</Text>
    </Pressable>
    {expanded && <>
      <Text style={ui.small}>Aim within the goal, as you face it. This records your intention.</Text>
      {[0, 3, 6].map(start => <View key={start} style={s.choices}>
        {targetZones.slice(start, start + 3).map(zone => <Choice key={zone} label={targetLabels[zone]}
          selected={intention.target_zone === zone} disabled={disabled}
          onPress={() => onChange({ ...intention, target_zone: intention.target_zone === zone ? null : zone })} />)}
      </View>)}
      <Choice label="No target selected" selected={intention.target_zone === null} disabled={disabled}
        onPress={() => onChange({ ...intention, target_zone: null })} />
      {includeOutcome && <>
        <Text style={ui.badge}>YOUR REPORTED OUTCOME</Text>
        <Text style={ui.small}>Your observation, not verified by SCAN.</Text>
        <View style={s.choices}>{reportedOutcomes.map(outcome => <Choice key={outcome} label={outcomeLabels[outcome]}
          selected={intention.reported_outcome === outcome} disabled={disabled}
          onPress={() => onChange({ ...intention, reported_outcome: intention.reported_outcome === outcome ? null : outcome })} />)}</View>
        <Choice label="Not reported" selected={intention.reported_outcome === null} disabled={disabled}
          onPress={() => onChange({ ...intention, reported_outcome: null })} />
      </>}
    </>}
  </View>;
}

const s = StyleSheet.create({
  choices: { flexDirection: 'row', gap: 7 },
  choice: { flex: 1, alignItems: 'center', justifyContent: 'center', minHeight: 44, borderRadius: 10,
    paddingVertical: 10, paddingHorizontal: 5, backgroundColor: colors.raised },
  selected: { backgroundColor: colors.green },
  label: { color: colors.text, fontSize: 12, fontWeight: '700', textAlign: 'center' },
});
