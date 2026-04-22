// ========================================
// Navbar Scroll Effect
// ========================================
window.addEventListener('scroll', function () {
    const navbar = document.getElementById('mainNav');
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ========================================
// Smooth Scrolling for Anchor Links
// ========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ========================================
// Active Navigation Link
// ========================================
const currentLocation = window.location.pathname;
const navLinks = document.querySelectorAll('.nav-link');

navLinks.forEach(link => {
    if (link.getAttribute('href') === currentLocation) {
        link.classList.add('active');
    }
});

// ========================================
// Loading Animation
// ========================================
window.addEventListener('load', function () {
    document.body.classList.add('loaded');
});

// ========================================
// Auto-hide Alerts
// ========================================
function autoHideAlert(alertId, duration = 5000) {
    const alert = document.getElementById(alertId);
    if (alert) {
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => {
                alert.style.display = 'none';
            }, 150);
        }, duration);
    }
}

// ========================================
// Form Validation Enhancement
// ========================================
(function () {
    'use strict';
    const forms = document.querySelectorAll('.needs-validation');

    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
})();

// ========================================
// Lazy Loading Images
// ========================================
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });

    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

// ========================================
// Back to Top Button
// ========================================
const backToTopButton = document.createElement('button');
backToTopButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
backToTopButton.className = 'back-to-top';
backToTopButton.style.cssText = `
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s;
    z-index: 1000;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
`;

document.body.appendChild(backToTopButton);

window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
        backToTopButton.style.opacity = '1';
        backToTopButton.style.visibility = 'visible';
    } else {
        backToTopButton.style.opacity = '0';
        backToTopButton.style.visibility = 'hidden';
    }
});

backToTopButton.addEventListener('click', () => {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
});

// ========================================
// Particle Background Effect (Optional)
// ========================================
function createParticles() {
    const particleContainer = document.querySelector('.hero-particles');
    if (!particleContainer) return;

    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(102, 126, 234, 0.5);
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: float ${5 + Math.random() * 10}s ease-in-out infinite;
        `;
        particleContainer.appendChild(particle);
    }
}

createParticles();

// ========================================
// Console Welcome Message
// ========================================
console.log('%c🔬 SkinCare AI Detection System', 'color: #667eea; font-size: 20px; font-weight: bold;');
console.log('%cBuilt with ❤️ using Flask, TensorFlow & Modern Web Technologies', 'color: #94a3b8; font-size: 12px;');
console.log('%cDeveloped by: Daksh Bhardwaj', 'color: #764ba2; font-size: 14px;');
