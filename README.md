# EZ-music-composition-bot
Much easier than the previous one

Only supports 1-5-6-3-4-1-2-5 chord progression

Only supports Verse(2 \* 8 bars)-Prechorus(8 bars)-Chorus(2 \* 8 bars) structure

Now the song is played 25% faster than before, with chords played together

# Added limits on LSTM output so that:
0. Only white keys (C D E F G A B) may occur, no sharps or flats

1. Any specific key will not occur too many times in a period (to provide diversity)

2. It is more likely that the note will fall on chords
(such as C,E,G are more likely to occur when C-Major chord is effective)
to make the song harmonious

3. From the 9th bar of each section, a note is more likely to have the same key as the note 8 bars ago.
(So the verse and chorus can be regarded as a 8-bar structure that repeats once more)

4. The pitch of a note is more likely to be around (<=9 semitones away from) both the previous one,
and the core (midi-number about 60 for verse, 66 for prechorus, 72 for chorus),
so that the note will not jump too far away from where it should be.
