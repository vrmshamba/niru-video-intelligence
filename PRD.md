# NIRU Video Intelligence Platform - MVP Product Requirements Document

**Version:** 1.0  
**Date:** December 30, 2024  
**Target:** NIRU Hackathon Demo (10 days)  
**Owner:** Mediatec Africa  

---

## Executive Summary

Build a working prototype that demonstrates AI-powered video intelligence for Kenyan content. The MVP proves core technology works using real Isiolo footage. System runs locally, processes 4 sample videos, and displays results on web dashboard.

---

## Problem Statement

Kenya lacks unified video intelligence infrastructure. Government agencies send untargeted messages. Broadcasters miss revenue opportunities. Safety teams respond late to harmful content. Planners lack video-based insights.

Current gaps:
- No automated content screening
- No speech-to-text for Kenyan languages
- No safe zones identified for message insertion
- No real-time topic tracking
- Scattered tools across agencies

---

## Success Metrics

### Hackathon Demo Success Criteria

1. Process 4 Isiolo videos successfully
2. Extract speech transcripts with 80%+ accuracy
3. Detect objects in video scenes
4. Flag at least 1 safety concern (if present in content)
5. Identify 3-5 safe message insertion points per video
6. Display all results on working dashboard
7. Complete demo in under 5 minutes

### Technical Success Metrics

- Processing time: Under 10 minutes per video
- Speech recognition: Swahili and English support
- Object detection: 80+ object types recognized
- Safety flagging: Weapons, violence, large crowds detected
- Dashboard load time: Under 3 seconds

---

## User Personas

### Primary: NIRU Hackathon Judges

**Goals:**
- Evaluate technical feasibility
- Assess real-world applicability
- Compare against other solutions

**Needs:**
- Clear demonstration of AI capabilities
- Proof of concept with real Kenyan content
- Understanding of scalability path

### Secondary: Government Agency Officer (Future User)

**Goals:**
- Monitor video content for safety
- Place public messages in appropriate content
- Track emerging topics across regions

**Needs:**
- Simple dashboard interface
- Real-time alerts for harmful content
- Clear insertion point recommendations

---

## Core Features (MVP Scope)

### Feature 1: Video Upload and Storage

**Description:** Accept video files and prepare for processing

**Requirements:**
- Accept MP4 format
- Support files up to 200MB
- Download from Google Drive
- Store locally in videos folder

**Out of Scope:**
- Live streaming support
- Multiple format conversion
- Cloud storage integration

### Feature 2: Speech-to-Text Analysis

**Description:** Extract spoken words from video audio

**Requirements:**
- Support Swahili and English
- Generate timestamped transcripts
- Detect language automatically
- Output full text and segments

**Technical Specs:**
- Model: Whisper Tiny
- Processing: CPU-based
- Accuracy target: 80%+
- Output format: JSON

**Out of Scope:**
- Real-time transcription
- Speaker identification
- Emotion detection

### Feature 3: Object and Scene Detection

**Description:** Identify objects, people, and scenes in video frames

**Requirements:**
- Scan video frames at regular intervals
- Detect people, vehicles, objects, backgrounds
- Tag each scene with detected items
- Generate confidence scores
- Create timestamp for each detection

**Technical Specs:**
- Model: YOLOv8n
- Sample rate: Every 30th frame (adjustable)
- Objects detected: 80+ types
- Confidence threshold: 50%+
- Output format: JSON with timestamps

**Out of Scope:**
- Facial recognition
- Emotion analysis
- Brand logo detection

### Feature 4: Safety Content Screening

**Description:** Automatically flag potentially harmful content

**Requirements:**
- Detect weapons (guns, knives)
- Flag violence indicators
- Identify large crowd gatherings (10+ people)
- Check transcript for sensitive keywords
- Generate alert with timestamp and description

**Safety Rules:**
- Dangerous objects: Immediate flag
- Large crowds: Medium severity flag
- Sensitive keywords: Context-based flag
- All flags logged with evidence

**Out of Scope:**
- Manual review workflow
- Alert routing to officers
- Content moderation actions

### Feature 5: Message Insertion Point Detection

**Description:** Find safe moments in video for message placement

**Requirements:**
- Identify stable scenes (low object movement)
- Avoid busy or chaotic moments
- Suggest 5-second insertion windows
- Rank by suitability
- Provide timestamp and duration

**Selection Criteria:**
- Scene stability (object count variation under 3)
- Low object density (under 5 objects)
- Minimum 5-second window
- Not adjacent to flagged content

**Out of Scope:**
- Actual message insertion/overlay
- Ad rendering
- Dynamic placement logic

### Feature 6: Web Dashboard

**Description:** Display all analysis results in visual interface

**Requirements:**
- Show summary statistics (total videos, flags, insertion points)
- Display individual video cards
- Show transcript for each video
- List detected objects and scenes
- Highlight safety concerns
- Display insertion point recommendations
- Responsive design for laptop screens

**Dashboard Sections:**
1. Header with platform title
2. Summary stats (3 cards)
3. Video analysis cards (grid layout)
4. Detailed view for each video

**Out of Scope:**
- Video playback in browser
- Real-time updates
- User authentication
- Multi-user support
- Export functionality

---

## Technical Architecture

### Technology Stack

**Backend:**
- Python 3.8+
- Flask (web framework)
- OpenCV (video processing)
- Whisper (speech-to-text)
- YOLOv8 (object detection)

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript
- No external frameworks

**Storage:**
- Local file system
- JSON for results
- No database required

**Hosting:**
- Local development server
- Port 5000 (Flask default)

### System Components

```
┌─────────────────────────────────────────────┐
│           User Interface Layer               │
│         (Web Dashboard - Browser)            │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│          Application Layer (Flask)           │
│  - API endpoints                             │
│  - Results aggregation                       │
│  - File serving                              │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│        Processing Layer (Python)             │
│  - Video intake                              │
│  - Whisper processing                        │
│  - YOLO detection                            │
│  - Safety analysis                           │
│  - Insertion point detection                 │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│           Storage Layer                      │
│  - videos/ (input files)                     │
│  - results/ (JSON output)                    │
│  - models/ (AI models)                       │
└─────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** Video files from Google Drive
2. **Processing:** 
   - Extract audio → Whisper → Transcript
   - Extract frames → YOLO → Object list
   - Combine results → Safety check
   - Analyze scenes → Insertion points
3. **Storage:** Save results as JSON
4. **Output:** Display on dashboard

### File Structure

```
niru-mvp/
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── download_videos.py        # Google Drive downloader
├── process_videos.py         # Main AI processing
├── app.py                    # Flask web server
├── videos/                   # Input videos
│   ├── Community gathering.mp4
│   ├── documentary isiolo.mp4
│   ├── news scene.mp4
│   └── Street scene.mp4
├── results/                  # Processing output
│   ├── Community gathering_analysis.json
│   ├── documentary isiolo_analysis.json
│   ├── news scene_analysis.json
│   ├── Street scene_analysis.json
│   └── processing_summary.json
├── models/                   # Downloaded AI models
│   ├── yolov8n.pt
│   └── whisper_tiny/
└── templates/                # HTML templates
    └── dashboard.html
```

---

## Data Schema

### Video Analysis Result (JSON)

```json
{
  "video_name": "news scene.mp4",
  "processed_at": "2024-12-30T14:30:00",
  "transcript": {
    "text": "Full transcript text...",
    "language": "sw",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "text": "Segment text..."
      }
    ]
  },
  "scene_analysis": {
    "total_scenes_analyzed": 120,
    "scenes": [
      {
        "timestamp": 2.5,
        "frame": 75,
        "objects": [
          {
            "object": "person",
            "confidence": 0.95
          }
        ]
      }
    ]
  },
  "safety_concerns": [
    {
      "type": "crowd_detection",
      "timestamp": 45.2,
      "description": "Large gathering detected: 15 people",
      "severity": "medium"
    }
  ],
  "insertion_points": [
    {
      "timestamp": 12.5,
      "duration": 5,
      "reason": "stable_scene",
      "object_count": 2
    }
  ],
  "summary": {
    "language_detected": "sw",
    "total_objects_detected": 450,
    "safety_flags": 2,
    "safe_insertion_windows": 5
  }
}
```

---

## Development Timeline

### Day 1 (Today): Setup and Foundation
- Create project folder
- Install dependencies
- Test Python environment
- Download sample videos
- Verify all files present

### Day 2: Core Processing Development
- Build video download script
- Implement Whisper integration
- Test speech extraction on 1 video
- Verify transcript quality

### Day 3: Object Detection
- Implement YOLO integration
- Process video frames
- Test object detection accuracy
- Optimize sample rate for performance

### Day 4: Safety and Analysis
- Build safety screening logic
- Implement insertion point detection
- Test flagging rules
- Validate detection accuracy

### Day 5: Process All Videos
- Run full pipeline on all 4 videos
- Generate JSON results
- Review output quality
- Fix any processing errors

### Day 6: Dashboard Development
- Build Flask application
- Create HTML template
- Implement API endpoints
- Test data display

### Day 7: Integration and Testing
- Connect dashboard to results
- Test full user flow
- Verify all features work
- Fix bugs

### Day 8: Polish and Demo Prep
- Improve dashboard design
- Add demo talking points
- Practice presentation flow
- Record backup video

### Day 9: Final Testing
- Run end-to-end test
- Verify on fresh environment
- Prepare deployment option
- Finalize presentation

### Day 10: Buffer and Rehearsal
- Final polish
- Rehearse demo
- Prepare for questions
- Deploy backup version

---

## Dependencies and Requirements

### Hardware Requirements

**Minimum:**
- Windows 10/11
- Intel i5 processor or equivalent
- 8GB RAM
- 5GB free disk space
- Internet connection (for setup only)

**Recommended:**
- Intel i7 processor or equivalent
- 16GB RAM
- 10GB free disk space

### Software Requirements

**Required:**
- Python 3.8 or higher
- VS Code or any text editor
- Web browser (Chrome, Firefox, Edge)
- Command line access

**Python Packages:**
- flask (web framework)
- opencv-python (video processing)
- openai-whisper (speech-to-text)
- torch (machine learning)
- ultralytics (YOLO)
- gdown (Google Drive download)
- moviepy (video manipulation)

---

## Out of Scope (Post-MVP Features)

### Not Included in Hackathon Demo

1. **Live Streaming Support**
   - Real-time video processing
   - Broadcaster feed integration
   - Continuous monitoring

2. **Actual Message Insertion**
   - Video overlay rendering
   - Dynamic ad placement
   - Server-side ad insertion (SSAI)

3. **Multi-User System**
   - User authentication
   - Role-based access
   - Agency dashboards

4. **Advanced Analytics**
   - Trend prediction
   - Pattern recognition
   - County-level aggregation

5. **Cloud Deployment**
   - Scalable infrastructure
   - Load balancing
   - CDN integration

6. **Mobile Applications**
   - Android app
   - iOS app
   - Mobile-optimized dashboard

7. **Revenue Systems**
   - Payment processing
   - Advertiser portal
   - Creator payouts

8. **Content Moderation Workflow**
   - Manual review interface
   - Approval workflows
   - Content blocking

---

## Risk Assessment

### Technical Risks

**Risk:** Processing takes too long on low-end CPU  
**Mitigation:** Use lightweight models, increase frame sampling interval, process overnight if needed

**Risk:** Speech recognition fails for Swahili content  
**Mitigation:** Whisper supports multiple languages, test early with Isiolo content, have English fallback

**Risk:** Videos don't download from Google Drive  
**Mitigation:** Manual download option provided, alternative hosting if needed

**Risk:** Dashboard doesn't load properly  
**Mitigation:** Simple HTML/CSS design, no complex dependencies, test early

### Timeline Risks

**Risk:** Processing all 4 videos takes longer than expected  
**Mitigation:** Start processing early (Day 2), run overnight, optimize sample rate

**Risk:** Last-minute bugs on demo day  
**Mitigation:** Complete testing by Day 8, record backup video, have offline demo ready

### Demo Risks

**Risk:** Internet fails during presentation  
**Mitigation:** Local hosting, no cloud dependencies, recorded backup video

**Risk:** Laptop crashes or freezes  
**Mitigation:** Test thoroughly, close unnecessary apps, have video backup

---

## Success Criteria

### Must Have (Required for Demo)

- [x] All 4 videos processed successfully
- [x] Dashboard loads and displays results
- [x] Speech transcripts visible and readable
- [x] Object detection results shown
- [x] At least 1 safety flag demonstrated
- [x] Insertion points identified
- [ ] Demo completes in under 5 minutes (Pending rehearsal)

### Should Have (Important but Not Critical)

- [x] Processing completes in under 40 minutes total
- [x] Transcript accuracy above 80% (Verified)
- [x] Clean, professional dashboard design (Updated with Premium UI)
- [ ] Recorded backup demo video (Pending)
- [x] Clear deployment path explained (In progress)

### Nice to Have (Bonus Points)

- [ ] Online deployed version (In progress)
- [ ] Sample message overlay demonstration
- [ ] Comparison with manual analysis
- [ ] Cost savings calculation
- [ ] Scalability roadmap presentation

---

## Testing Plan

### Unit Testing

- Whisper transcription on single video
- YOLO detection on single frame
- Safety flag logic with test data
- Insertion point algorithm validation

### Integration Testing

- Full pipeline on 1 video
- Dashboard display of results
- API endpoint responses
- File reading and writing

### End-to-End Testing

- Process all 4 videos
- Load dashboard
- Navigate all features
- Verify data accuracy
- Check performance

### Demo Rehearsal

- Run complete demo flow
- Time presentation (under 5 minutes)
- Practice Q&A responses
- Test on fresh environment

---

## Maintenance and Support

### Post-Hackathon Actions

1. Collect judge feedback
2. Document lessons learned
3. Identify improvement areas
4. Plan next development phase

### Future Development Path

**Phase 2 (Weeks 1-4):**
- Add actual message insertion
- Deploy to cloud platform
- Onboard 1 test broadcaster

**Phase 3 (Months 2-3):**
- Build agency control panel
- Add live streaming support
- Implement user authentication

**Phase 4 (Months 4-6):**
- Scale to multiple broadcasters
- Add revenue tracking
- Launch commercial pilot

---

## Appendix

### Glossary

**Computer Vision:** AI that understands images and video  
**YOLO:** Real-time object detection model  
**Whisper:** Speech recognition AI by OpenAI  
**Flask:** Python web framework  
**MVP:** Minimum Viable Product  
**SSAI:** Server-Side Ad Insertion  

### References

- Whisper Documentation: https://github.com/openai/whisper
- YOLOv8 Documentation: https://docs.ultralytics.com
- Flask Documentation: https://flask.palletsprojects.com
- OpenCV Documentation: https://docs.opencv.org

### Contact

**Project Owner:** Mediatec Africa  
**Timeline:** 10 days (Dec 30, 2024 - Jan 8, 2025)  
**Demo Date:** Day 10  

---

**Document Status:** Final  
**Approved for Development:** Yes  
**Next Action:** Begin Day 1 setup and installation