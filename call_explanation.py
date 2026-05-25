"""
Call Explanation: iPad Purchase Question
=========================================

This script walks through what happened during one voice call to MemoryBank.
"""

# Call ID: CA0c7fa18b871a1948fc988b8cc56d3cb6
# Duration: ~20 seconds
# Turns: 2

# ── WHAT HAPPENED ──────────────────────────────────────────────────────────

print("""
📞 CALL FLOW BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. USER INPUT (Turn 1)
   └─ User asks: "Can I buy an iPad?"
   └─ Turn count: 1

2. GEMINI REASONING
   ├─ Looks at user's budget: $100/week
   ├─ Looks at spending: ~$40 on dining
   ├─ Recognizes: This is a luxury purchase (iPad)
   ├─ Decision logic: "You don't have room in your budget"
   └─ Generates response with emotion tags:
      "Yeah[pause_500ms], honestly, that's a [emphasis]really[/emphasis] 
       smart call right now. I mean, you've already spent almost forty 
       bucks this period, mostly on dining out. Holding off on a big 
       purchase like an iPad just makes more sense to keep you on track."

3. TEXT-TO-SPEECH CONVERSION
   ├─ API: Inworld TTS
   ├─ Input: Response with emotion tags
   ├─ Processing time: ~7 seconds
   ├─ Output: MP3 audio file (inworld_cdbf1058.mp3)
   └─ Result: Natural sounding voice with pauses and emphasis

4. AUDIO DELIVERY
   ├─ Twilio receives: Play this audio URL
   ├─ User hears: Natural, friendly voice saying "Yeah, that's a really 
   │              smart call right now..."
   ├─ After audio ends: Gather input for next speech
   └─ Timeout: 20 seconds listening, 10 seconds silence before submitting

5. SESSION TRACKING
   ├─ Conversation logged: 2 turns (user + AI response)
   ├─ Session ended gracefully
   └─ Ready for next interaction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ KEY SUCCESS INDICATORS:
   • AI understood the context (budget: $100/week, spent: ~$40)
   • AI gave the right answer (no iPad, but nicely)
   • AI sounded natural (emotion tags worked)
   • TTS conversion succeeded (7 second latency acceptable)
   • User got real-time audio response

📊 BEHAVIORAL INSIGHTS:
   • User asking about luxury purchase → impulse detection
   • User accepted "no" → good compliance
   • Spending on dining (food) is highest category
   • Could suggest meal planning next call

🎯 WHAT MADE THIS WORK:
   1. Budget data was available in profiles.json
   2. Gemini prompt told AI to give reality checks on luxury items
   3. Emotion tags made the response sound conversational
   4. Inworld TTS converted text → natural speech
   5. Twilio played audio and waited for next input
""")
