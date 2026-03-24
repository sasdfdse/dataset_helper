/**
 * VideoDatasetTool - Qt5 GUI
 * Build: qmake video_tool_qt.pro && make
 */

#include "video_core.hpp"

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QFileDialog>
#include <QThread>
#include <QGroupBox>
#include <QMessageBox>
#include <QStatusBar>

// ═══════════════════════════════════════════════════════════════════
// Qt Workers
// ═══════════════════════════════════════════════════════════════════

class ConvertWorker : public QObject {
    Q_OBJECT
public:
    QString inputPath, outputPath;
signals:
    void progress(int pct);
    void finished(bool ok, QString msg, double elapsed, qint64 size);
public slots:
    void run() {
        auto r = convertWebmToMp4(
            inputPath.toStdString(), outputPath.toStdString(),
            [this](int pct, const std::string&) { emit progress(pct); });
        emit finished(r.success, QString::fromStdString(r.message),
                      r.elapsedSec, (qint64)r.outputSize);
    }
};

class ExtractWorker : public QObject {
    Q_OBJECT
public:
    QString videoPath;
    ExtractOptions opts;
signals:
    void progress(int pct);
    void finished(bool ok, QString msg, int total, int saved, double elapsed);
public slots:
    void run() {
        auto r = extractFrames(
            videoPath.toStdString(), opts,
            [this](int pct, const std::string&) { emit progress(pct); });
        emit finished(r.success, QString::fromStdString(r.message),
                      r.totalFrames, r.savedFrames, r.elapsedSec);
    }
};

class ConvertExtractWorker : public QObject {
    Q_OBJECT
public:
    QString inputPath, mp4Path;
    ExtractOptions extractOpts;
signals:
    void progress(int pct);
    void log(QString msg);
    void finished(bool ok, QString msg, int saved, double elapsed);
public slots:
    void run() {
        emit log("[1/2] WEBM → MP4 변환 중...");
        auto cr = convertWebmToMp4(
            inputPath.toStdString(), mp4Path.toStdString(),
            [this](int pct, const std::string&) { emit progress(pct / 2); });
        if (!cr.success) {
            emit finished(false, "변환 실패: " + QString::fromStdString(cr.message), 0, 0);
            return;
        }
        emit log(QString("✓ MP4 변환 완료 (%1)")
                 .arg(QString::fromStdString(humanSize(cr.outputSize))));

        emit log("[2/2] 프레임 추출 중...");
        auto er = extractFrames(
            mp4Path.toStdString(), extractOpts,
            [this](int pct, const std::string&) { emit progress(50 + pct / 2); });
        if (!er.success) {
            emit finished(false, "추출 실패: " + QString::fromStdString(er.message), 0, 0);
            return;
        }
        emit finished(true, "", er.savedFrames, cr.elapsedSec + er.elapsedSec);
    }
};

// ═══════════════════════════════════════════════════════════════════
// UI Helpers
// ═══════════════════════════════════════════════════════════════════

static QWidget* makeFileRow(QLineEdit** lineOut, const QString& label,
                              bool isSave, const QString& filter, QWidget* parent) {
    QWidget* row = new QWidget(parent);
    QHBoxLayout* hl = new QHBoxLayout(row);
    hl->setContentsMargins(0, 0, 0, 0);
    QLabel* lbl = new QLabel(label, row);
    lbl->setFixedWidth(150);
    QLineEdit* le = new QLineEdit(row);
    QPushButton* btn = new QPushButton("...", row);
    btn->setFixedWidth(36);
    hl->addWidget(lbl); hl->addWidget(le); hl->addWidget(btn);
    *lineOut = le;

    QObject::connect(btn, &QPushButton::clicked, [le, isSave, filter]() {
        QString path = isSave
            ? QFileDialog::getSaveFileName(nullptr, "저장 경로 선택", le->text(), filter)
            : QFileDialog::getOpenFileName(nullptr, "파일 선택", le->text(), filter);
        if (!path.isEmpty()) le->setText(path);
    });
    return row;
}

static QWidget* makeDirRow(QLineEdit** lineOut, const QString& label, QWidget* parent) {
    QWidget* row = new QWidget(parent);
    QHBoxLayout* hl = new QHBoxLayout(row);
    hl->setContentsMargins(0, 0, 0, 0);
    QLabel* lbl = new QLabel(label, row);
    lbl->setFixedWidth(150);
    QLineEdit* le = new QLineEdit(row);
    QPushButton* btn = new QPushButton("...", row);
    btn->setFixedWidth(36);
    hl->addWidget(lbl); hl->addWidget(le); hl->addWidget(btn);
    *lineOut = le;

    QObject::connect(btn, &QPushButton::clicked, [le]() {
        QString path = QFileDialog::getExistingDirectory(nullptr, "디렉토리 선택", le->text());
        if (!path.isEmpty()) le->setText(path);
    });
    return row;
}

// ═══════════════════════════════════════════════════════════════════
// Tab 1: WEBM → MP4
// ═══════════════════════════════════════════════════════════════════
class ConvertTab : public QWidget {
    Q_OBJECT
    QLineEdit*   inputEdit;
    QLineEdit*   outputEdit;
    QPushButton* startBtn;
    QProgressBar* progressBar;
    QTextEdit*   logEdit;
    QThread*     thread = nullptr;
public:
    ConvertTab(QWidget* parent = nullptr) : QWidget(parent) {
        QVBoxLayout* layout = new QVBoxLayout(this);

        QGroupBox* grp = new QGroupBox("WEBM → MP4 변환", this);
        QVBoxLayout* gl = new QVBoxLayout(grp);
        gl->addWidget(makeFileRow(&inputEdit,  "입력 WEBM 파일:", false,
                                  "Video (*.webm *.mkv *.avi *.mp4 *.*);;All (*)", grp));
        gl->addWidget(makeFileRow(&outputEdit, "출력 MP4 파일:", true,
                                  "MP4 (*.mp4)", grp));
        connect(inputEdit, &QLineEdit::textChanged, [this](const QString& t) {
            if (!t.isEmpty() && outputEdit->text().isEmpty()) {
                fs::path p(t.toStdString());
                outputEdit->setText(
                    QString::fromStdString((p.parent_path() / p.stem()).string() + ".mp4"));
            }
        });
        layout->addWidget(grp);

        startBtn = new QPushButton("▶  변환 시작", this);
        startBtn->setFixedHeight(38);
        layout->addWidget(startBtn);

        progressBar = new QProgressBar(this);
        progressBar->setRange(0, 100);
        layout->addWidget(progressBar);

        logEdit = new QTextEdit(this);
        logEdit->setReadOnly(true);
        logEdit->setFixedHeight(140);
        layout->addWidget(logEdit);

        layout->addStretch();
        connect(startBtn, &QPushButton::clicked, this, &ConvertTab::onStart);
    }

private slots:
    void onStart() {
        QString inp = inputEdit->text().trimmed();
        QString out = outputEdit->text().trimmed();
        if (inp.isEmpty() || out.isEmpty()) {
            QMessageBox::warning(this, "경고", "입력/출력 경로를 입력해주세요."); return;
        }
        startBtn->setEnabled(false);
        progressBar->setValue(0);
        logEdit->clear();
        logEdit->append("변환 시작...");

        thread = new QThread(this);
        ConvertWorker* worker = new ConvertWorker();
        worker->inputPath  = inp;
        worker->outputPath = out;
        worker->moveToThread(thread);

        connect(thread, &QThread::started, worker, &ConvertWorker::run);
        connect(worker, &ConvertWorker::progress, progressBar, &QProgressBar::setValue);
        connect(worker, &ConvertWorker::finished, this,
                [this, out](bool ok, QString msg, double elapsed, qint64 size) {
            if (ok) {
                progressBar->setValue(100);
                logEdit->append(QString("✓ 변환 완료!\n  출력: %1\n  크기: %2\n  소요: %3초")
                    .arg(out)
                    .arg(QString::fromStdString(humanSize((uintmax_t)size)))
                    .arg(elapsed, 0, 'f', 1));
            } else {
                logEdit->append("✗ 오류: " + msg);
            }
            startBtn->setEnabled(true);
            thread->quit();
        });
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        thread->start();
    }
};

// ═══════════════════════════════════════════════════════════════════
// Tab 2: Frame Extraction
// ═══════════════════════════════════════════════════════════════════
class ExtractTab : public QWidget {
    Q_OBJECT
    QLineEdit*    inputEdit;
    QLineEdit*    outputDirEdit;
    QComboBox*    ratioCombo;
    QCheckBox*    grayCheck;
    QPushButton*  startBtn;
    QProgressBar* progressBar;
    QTextEdit*    logEdit;
    QThread*      thread = nullptr;
public:
    ExtractTab(QWidget* parent = nullptr) : QWidget(parent) {
        QVBoxLayout* layout = new QVBoxLayout(this);

        QGroupBox* grp = new QGroupBox("프레임 추출", this);
        QVBoxLayout* gl = new QVBoxLayout(grp);
        gl->addWidget(makeFileRow(&inputEdit, "입력 동영상:", false,
                                  "Video (*.mp4 *.webm *.mkv *.avi *.*);;All (*)", grp));
        gl->addWidget(makeDirRow(&outputDirEdit, "출력 디렉토리:", grp));
        connect(inputEdit, &QLineEdit::textChanged, [this](const QString& t) {
            if (!t.isEmpty() && outputDirEdit->text().isEmpty()) {
                outputDirEdit->setText(
                    QString::fromStdString(fs::path(t.toStdString()).stem().string()) + "_frames");
            }
        });

        QWidget* ratioRow = new QWidget(grp);
        QHBoxLayout* rh = new QHBoxLayout(ratioRow);
        rh->setContentsMargins(0, 0, 0, 0);
        QLabel* rl = new QLabel("추출 비율:", ratioRow);
        rl->setFixedWidth(150);
        ratioCombo = new QComboBox(ratioRow);
        ratioCombo->addItem("전체 (1/1)", 1);
        ratioCombo->addItem("절반 (1/2)", 2);
        ratioCombo->addItem("3분의 1 (1/3)", 3);
        rh->addWidget(rl); rh->addWidget(ratioCombo); rh->addStretch();
        gl->addWidget(ratioRow);

        grayCheck = new QCheckBox("그레이스케일로 저장", grp);
        gl->addWidget(grayCheck);
        layout->addWidget(grp);

        startBtn = new QPushButton("▶  추출 시작", this);
        startBtn->setFixedHeight(38);
        layout->addWidget(startBtn);

        progressBar = new QProgressBar(this);
        progressBar->setRange(0, 100);
        layout->addWidget(progressBar);

        logEdit = new QTextEdit(this);
        logEdit->setReadOnly(true);
        logEdit->setFixedHeight(140);
        layout->addWidget(logEdit);

        layout->addStretch();
        connect(startBtn, &QPushButton::clicked, this, &ExtractTab::onStart);
    }

private slots:
    void onStart() {
        QString inp    = inputEdit->text().trimmed();
        QString outDir = outputDirEdit->text().trimmed();
        if (inp.isEmpty() || outDir.isEmpty()) {
            QMessageBox::warning(this, "경고", "입력 파일과 출력 디렉토리를 입력해주세요."); return;
        }
        startBtn->setEnabled(false);
        progressBar->setValue(0);
        logEdit->clear();
        logEdit->append("프레임 추출 시작...");

        ExtractOptions opts;
        opts.ratio     = ratioCombo->currentData().toInt();
        opts.grayscale = grayCheck->isChecked();
        opts.format    = "jpg";
        opts.outDir    = outDir.toStdString();

        thread = new QThread(this);
        ExtractWorker* worker = new ExtractWorker();
        worker->videoPath = inp;
        worker->opts      = opts;
        worker->moveToThread(thread);

        connect(thread, &QThread::started, worker, &ExtractWorker::run);
        connect(worker, &ExtractWorker::progress, progressBar, &QProgressBar::setValue);
        connect(worker, &ExtractWorker::finished, this,
                [this, outDir](bool ok, QString msg, int total, int saved, double elapsed) {
            if (ok) {
                progressBar->setValue(100);
                logEdit->append(
                    QString("✓ 추출 완료!\n  총 프레임: %1\n  저장된 이미지: %2장\n  출력: %3\n  소요: %4초")
                    .arg(total).arg(saved).arg(outDir).arg(elapsed, 0, 'f', 1));
            } else {
                logEdit->append("✗ 오류: " + msg);
            }
            startBtn->setEnabled(true);
            thread->quit();
        });
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        thread->start();
    }
};

// ═══════════════════════════════════════════════════════════════════
// Tab 3: Convert + Extract
// ═══════════════════════════════════════════════════════════════════
class ConvertExtractTab : public QWidget {
    Q_OBJECT
    QLineEdit*    inputEdit;
    QLineEdit*    mp4Edit;
    QLineEdit*    framesEdit;
    QComboBox*    ratioCombo;
    QCheckBox*    grayCheck;
    QPushButton*  startBtn;
    QProgressBar* progressBar;
    QTextEdit*    logEdit;
    QThread*      thread = nullptr;
public:
    ConvertExtractTab(QWidget* parent = nullptr) : QWidget(parent) {
        QVBoxLayout* layout = new QVBoxLayout(this);

        QGroupBox* grp = new QGroupBox("WEBM → MP4 변환 + 프레임 추출 (연속 작업)", this);
        QVBoxLayout* gl = new QVBoxLayout(grp);
        gl->addWidget(makeFileRow(&inputEdit, "입력 WEBM 파일:", false,
                                  "Video (*.webm *.*);;All (*)", grp));
        gl->addWidget(makeFileRow(&mp4Edit,   "MP4 출력 경로:", true,
                                  "MP4 (*.mp4)", grp));
        gl->addWidget(makeDirRow(&framesEdit, "프레임 디렉토리:", grp));

        connect(inputEdit, &QLineEdit::textChanged, [this](const QString& t) {
            if (t.isEmpty()) return;
            fs::path p(t.toStdString());
            if (mp4Edit->text().isEmpty())
                mp4Edit->setText(
                    QString::fromStdString((p.parent_path() / p.stem()).string() + ".mp4"));
            if (framesEdit->text().isEmpty())
                framesEdit->setText(
                    QString::fromStdString(p.stem().string()) + "_frames");
        });

        QWidget* ratioRow = new QWidget(grp);
        QHBoxLayout* rh = new QHBoxLayout(ratioRow);
        rh->setContentsMargins(0, 0, 0, 0);
        QLabel* rl = new QLabel("추출 비율:", ratioRow);
        rl->setFixedWidth(150);
        ratioCombo = new QComboBox(ratioRow);
        ratioCombo->addItem("전체 (1/1)", 1);
        ratioCombo->addItem("절반 (1/2)", 2);
        ratioCombo->addItem("3분의 1 (1/3)", 3);
        rh->addWidget(rl); rh->addWidget(ratioCombo); rh->addStretch();
        gl->addWidget(ratioRow);

        grayCheck = new QCheckBox("그레이스케일로 저장", grp);
        gl->addWidget(grayCheck);
        layout->addWidget(grp);

        startBtn = new QPushButton("▶  작업 시작", this);
        startBtn->setFixedHeight(38);
        layout->addWidget(startBtn);

        progressBar = new QProgressBar(this);
        progressBar->setRange(0, 100);
        layout->addWidget(progressBar);

        logEdit = new QTextEdit(this);
        logEdit->setReadOnly(true);
        logEdit->setFixedHeight(160);
        layout->addWidget(logEdit);

        layout->addStretch();
        connect(startBtn, &QPushButton::clicked, this, &ConvertExtractTab::onStart);
    }

private slots:
    void onStart() {
        QString inp    = inputEdit->text().trimmed();
        QString mp4    = mp4Edit->text().trimmed();
        QString frames = framesEdit->text().trimmed();
        if (inp.isEmpty() || mp4.isEmpty() || frames.isEmpty()) {
            QMessageBox::warning(this, "경고", "모든 경로를 입력해주세요."); return;
        }
        startBtn->setEnabled(false);
        progressBar->setValue(0);
        logEdit->clear();

        ExtractOptions opts;
        opts.ratio     = ratioCombo->currentData().toInt();
        opts.grayscale = grayCheck->isChecked();
        opts.format    = "jpg";
        opts.outDir    = frames.toStdString();

        thread = new QThread(this);
        ConvertExtractWorker* worker = new ConvertExtractWorker();
        worker->inputPath    = inp;
        worker->mp4Path      = mp4;
        worker->extractOpts  = opts;
        worker->moveToThread(thread);

        connect(thread, &QThread::started, worker, &ConvertExtractWorker::run);
        connect(worker, &ConvertExtractWorker::progress, progressBar, &QProgressBar::setValue);
        connect(worker, &ConvertExtractWorker::log, logEdit, &QTextEdit::append);
        connect(worker, &ConvertExtractWorker::finished, this,
                [this, frames](bool ok, QString msg, int saved, double elapsed) {
            if (ok) {
                progressBar->setValue(100);
                logEdit->append(
                    QString("✓ 모든 작업 완료!\n  저장된 이미지: %1장\n  출력: %2\n  총 소요: %3초")
                    .arg(saved).arg(frames).arg(elapsed, 0, 'f', 1));
            } else {
                logEdit->append("✗ 오류: " + msg);
            }
            startBtn->setEnabled(true);
            thread->quit();
        });
        connect(thread, &QThread::finished, worker, &QObject::deleteLater);
        connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        thread->start();
    }
};

// ═══════════════════════════════════════════════════════════════════
// Tab 4: Video Info
// ═══════════════════════════════════════════════════════════════════
class InfoTab : public QWidget {
    Q_OBJECT
    QLineEdit*   inputEdit;
    QPushButton* queryBtn;
    QTextEdit*   infoEdit;
public:
    InfoTab(QWidget* parent = nullptr) : QWidget(parent) {
        QVBoxLayout* layout = new QVBoxLayout(this);

        QGroupBox* grp = new QGroupBox("동영상 정보", this);
        QVBoxLayout* gl = new QVBoxLayout(grp);
        gl->addWidget(makeFileRow(&inputEdit, "동영상 파일:", false,
                                  "Video (*.mp4 *.webm *.mkv *.avi *.*);;All (*)", grp));
        layout->addWidget(grp);

        queryBtn = new QPushButton("🔍  정보 조회", this);
        queryBtn->setFixedHeight(38);
        layout->addWidget(queryBtn);

        infoEdit = new QTextEdit(this);
        infoEdit->setReadOnly(true);
        infoEdit->setFontFamily("Monospace");
        layout->addWidget(infoEdit);

        connect(queryBtn, &QPushButton::clicked, this, &InfoTab::onQuery);
    }

private slots:
    void onQuery() {
        QString inp = inputEdit->text().trimmed();
        if (inp.isEmpty()) return;
        VideoInfo info = getVideoInfo(inp.toStdString());
        if (!info.valid) {
            infoEdit->setText("오류: " + QString::fromStdString(info.error)); return;
        }
        QString text;
        text += QString("파일:        %1\n").arg(inp);
        text += QString("포맷:        %1\n").arg(QString::fromStdString(info.format));
        text += QString("길이:        %1분 %2초\n").arg(info.durationSec / 60).arg(info.durationSec % 60);
        if (info.bitrate > 0)
            text += QString("비트레이트:  %1 kbps\n").arg(info.bitrate / 1000);
        if (!info.videoCodec.empty()) {
            text += "\n[비디오]\n";
            text += QString("  코덱:      %1\n").arg(QString::fromStdString(info.videoCodec));
            text += QString("  해상도:    %1 × %2\n").arg(info.width).arg(info.height);
            if (info.fps > 0)
                text += QString("  FPS:       %1\n").arg(info.fps, 0, 'f', 2);
            if (info.nbFrames > 0)
                text += QString("  총 프레임: %1\n").arg(info.nbFrames);
        }
        if (!info.audioCodec.empty()) {
            text += "\n[오디오]\n";
            text += QString("  코덱:      %1\n").arg(QString::fromStdString(info.audioCodec));
            text += QString("  샘플레이트: %1 Hz\n").arg(info.sampleRate);
            text += QString("  채널:      %1\n").arg(info.channels);
        }
        infoEdit->setText(text);
    }
};

// ═══════════════════════════════════════════════════════════════════
// MainWindow
// ═══════════════════════════════════════════════════════════════════
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("VideoDatasetTool v1.0");
        setMinimumSize(640, 560);

        QTabWidget* tabs = new QTabWidget(this);
        tabs->addTab(new ConvertTab(tabs),        "WEBM → MP4");
        tabs->addTab(new ExtractTab(tabs),         "프레임 추출");
        tabs->addTab(new ConvertExtractTab(tabs),  "변환 + 추출");
        tabs->addTab(new InfoTab(tabs),            "동영상 정보");
        setCentralWidget(tabs);
        statusBar()->showMessage("준비");
    }
};

// ─── main ─────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    av_log_set_level(AV_LOG_ERROR);
    QApplication app(argc, argv);
    app.setStyle("Fusion");
    MainWindow win;
    win.show();
    return app.exec();
}

#include "dataset_helper_qt.moc"
