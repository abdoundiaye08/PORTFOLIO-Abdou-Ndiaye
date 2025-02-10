// Ajouter un effet de défilement fluide pour les liens de navigation
document.querySelectorAll('nav ul li a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Ajouter un effet de survol pour les images de la galerie
document.querySelectorAll('.gallery img').forEach(img => {
    img.addEventListener('mouseover', () => {
        img.style.transform = 'scale(1.05)';
        img.style.transition = 'transform 0.3s ease';
    });
    img.addEventListener('mouseout', () => {
        img.style.transform = 'scale(1)';
    });
});